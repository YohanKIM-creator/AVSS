import torch
import torch.nn as nn
import torch.nn.functional as F
# Before Writing Models we should write audio preprocessing script and audio augmentation

# input shape -> [B, 1, T]
class AudioEncoder(nn.Module):
    # Encode mono waveforms from [B, 1, N] to [B, F0, T].
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride),
            nn.GELU()
        )

    def forward(self, x):
        # uncomment if use librosa.load() instead of torchaudio.load()
        # x = torch.unsqueeze(x, dim=0) if len(x.shape) == 1 else torch.unsqueeze(x, dim=1)
        return self.audio_encoder(x) # [B, 1, N] -> [B, C0, T]

class RotaryAttention(nn.Module):
    def __init__(self, embed_dim, heads, dropout, radius=None):
        super().__init__()
        self.heads = heads
        self.head_dim = embed_dim // heads
        self.dropout = dropout
        self.radius = radius
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        inv = 10000.0**(-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inv_freq", inv, persistent=False)

    # x.shape = [B, H, T, D]
    def rotate(self, x):
        phase = torch.outer(torch.arange(x.shape[-2], device=x.device).float(), self.inv_freq.float())
        cos, sin = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def local_attention(self, q, k, v, dropout):
        # instead of Convolution Local Attention we use Local Attention with sliding window
        B, H, T, D = q.shape
        block = min(2 * self.radius, T)
        blocks = (T + block - 1) // block
        extra = blocks * block - T
        width = block + 2 * self.radius

        q = F.pad(q, (0, 0, 0, extra)).reshape(B * H, blocks, block, D)
        k = F.pad(k, (0, 0, self.radius, self.radius + extra))
        v = F.pad(v, (0, 0, self.radius, self.radius + extra))
        k = k.unfold(2, width, block).transpose(-1, -2).reshape(B * H, blocks, width, D)
        v = v.unfold(2, width, block).transpose(-1, -2).reshape(B * H, blocks, width, D)

        starts = block * torch.arange(blocks, device=q.device)[:, None]
        query_pos = (starts + torch.arange(block, device=q.device)).clamp_max(T - 1)
        key_pos = starts + torch.arange(width, device=q.device) - self.radius
        distance = (query_pos[:, :, None] - key_pos[:, None, :]).abs()
        valid_keys = (key_pos >= 0) & (key_pos < T)
        mask = (distance <= self.radius) & valid_keys[:, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout)

        return y.reshape(B, H, blocks * block, D)[:, :, :T]

    # x.shape = [B, T, C]
    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = q.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.heads, self.head_dim).transpose(1, 2)
        q = self.rotate(q)
        k = self.rotate(k)

        if self.radius is None:
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        else:
            y = self.local_attention(q, k, v, self.dropout)

        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(y)

class GlobalTransformer(nn.Module):
    # we use Efficient Global Attention (EGA) with RoPE instead of RPE
    def __init__(self, dim, heads, kernel_size, dropout):
        super().__init__()
        self.down_sample = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, ceil_mode=True)
        self.mhsa = RotaryAttention(dim, heads, dropout=0.0)
        self.upsample = nn.Upsample(scale_factor=kernel_size, mode="nearest")
        self.gate = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pooled = self.down_sample(x.transpose(1, 2)).transpose(1, 2)
        y = self.mhsa(pooled)
        y = self.upsample(y.transpose(1, 2)).transpose(1, 2)
        y = y[:, :x.size(1)]
        return self.dropout(y * self.gate(x).sigmoid())

class LocalTransformer(nn.Module):
    # Local block adapted to use windowed RoPE attention
    def __init__(self, dim, heads, radius=32, dropout=0.1):
        super().__init__()
        self.local_attention = nn.Sequential(
            RotaryAttention(dim, heads, dropout=0.0, radius=radius),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.local_attention(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, mixer, dropout, layer_scale_init):
        super().__init__()
        self.mix_norm = nn.LayerNorm(dim)
        self.mixer = mixer
        self.mix_scale = nn.Parameter(torch.full((dim,), layer_scale_init))
        self.ffn_norm = nn.LayerNorm(dim)
        self.gcfn = GCFN(dim, dropout)
        self.ffn_scale = nn.Parameter(torch.full((dim,), layer_scale_init))

    def forward(self, x):
        x = x + self.mix_scale * self.mixer(self.mix_norm(x))
        return x + self.ffn_scale * self.gcfn(self.ffn_norm(x))

class GCFN(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.in_proj = nn.Linear(dim, 6 * dim)
        self.depthwise = nn.Conv1d(6 * dim, 6 * dim, kernel_size=3, padding=1, groups=6 * dim)
        self.out_proj = nn.Linear(3 * dim, dim) # equivalent to pointwise convolution
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = [B, T, C]
        x = self.in_proj(x)
        x = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(F.glu(x, dim=-1))
        return self.dropout(self.out_proj(x))

class SeparationEncoder(nn.Module):
    # Separation encoder, including the bottleneck stage
    def __init__(self, audio_dim, dim, heads, levels, radius, dropout, layer_scale_init):
        super().__init__()
        # Input layer: project each frame from F0 to F.
        self.input_layer = nn.Sequential(nn.Linear(audio_dim, dim), nn.LayerNorm(dim))
        self.stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        for level in range(levels + 1):
            # EGA attends at the bottleneck length at every resolution
            pool = 2 ** (levels - level)

            # B_E = 2: two Global -> Local pairs per encoder level
            global1 = GlobalTransformer(dim, heads, pool, dropout)
            local1 = LocalTransformer(dim, heads, radius, dropout)
            global2 = GlobalTransformer(dim, heads, pool, dropout)
            local2 = LocalTransformer(dim, heads, radius, dropout)

            self.stages.append(nn.Sequential(
                TransformerBlock(dim, global1, dropout, layer_scale_init),
                TransformerBlock(dim, local1, dropout, layer_scale_init),
                TransformerBlock(dim, global2, dropout, layer_scale_init),
                TransformerBlock(dim, local2, dropout, layer_scale_init)
            ))

            # Down Conv uses a depthwise stride-2 convolution, BN, and GELU
            if level < levels:
                self.downs.append(nn.Sequential(
                    nn.Conv1d(dim, dim, 5, stride=2, padding=2, groups=dim),
                    nn.BatchNorm1d(dim),
                    nn.GELU()
                ))

    def forward(self, x):
        x = self.input_layer(x.transpose(1, 2))  # [B, T, F]
        skips = []

        for stage, down in zip(self.stages[:-1], self.downs):
            x = stage(x)
            # Save features at the current resolution for the decoder
            skips.append(x)
            x = down(x.transpose(1, 2)).transpose(1, 2)

        return self.stages[-1](x), skips


class DecoderBlock(nn.Module):
    # One Global -> Local -> Cross-Speaker
    def __init__(self, dim, heads, pool, radius, dropout, layer_scale_init):
        super().__init__()
        self.global_block = TransformerBlock(dim, GlobalTransformer(dim, heads, pool, dropout), dropout, layer_scale_init)
        self.local_block = TransformerBlock(dim, LocalTransformer(dim, heads, radius, dropout), dropout, layer_scale_init)
        self.cross_speaker = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

    def forward(self, x):
        B, J, T, C = x.shape
        # Fold speakers into the batch so they share the temporal blocks.
        x = x.reshape(B * J, T, C)
        x = self.local_block(self.global_block(x))

        # Attend across speakers independently at each frame.
        x = x.reshape(B, J, T, C).transpose(1, 2).reshape(B * T, J, C)
        x = self.cross_speaker(x)
        return x.reshape(B, T, J, C).transpose(1, 2)


class ReconstructionStage(nn.Module):
    def __init__(self, dim, heads, pool, radius, dropout, layer_scale_init):
        super().__init__()
        self.merge = nn.Linear(2 * dim, dim)

        # B_D = 3: independent Global -> Local -> Cross-Speaker repeats.
        self.block1 = DecoderBlock(dim, heads, pool, radius, dropout, layer_scale_init)
        self.block2 = DecoderBlock(dim, heads, pool, radius, dropout, layer_scale_init)
        self.block3 = DecoderBlock(dim, heads, pool, radius, dropout, layer_scale_init)

    def forward(self, x, skip):
        # nearest upsampling, skip concatenation, then a linear merge.
        x = x.repeat_interleave(2, dim=2)
        x = self.merge(torch.cat((x, skip), dim=-1))
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)


class Separator(nn.Module):
    def __init__(self, audio_dim, dim, heads, levels, speakers, radius, dropout, layer_scale_init):
        super().__init__()
        self.speakers = speakers
        self.encoder = SeparationEncoder(audio_dim, dim, heads, levels, radius, dropout, layer_scale_init)

        # Fig. 3: the first Linear packs both F -> 2JF branches.
        split_hidden = 2 * speakers * dim
        self.split = nn.Sequential(
            nn.Linear(dim, 2 * split_hidden),
            nn.GLU(),
            nn.Linear(split_hidden, speakers * dim)
        )
        self.split_norm = nn.LayerNorm(dim)

        self.decoder = nn.ModuleList(ReconstructionStage(dim, heads, 2 ** level, radius, dropout, layer_scale_init) for level in range(1, levels + 1))

        # Project each speaker's features back to the audio encoder width F0
        self.output = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GLU(), nn.Linear(dim, audio_dim))

    def split_speakers(self, x):
        B, T, C = x.shape
        x = self.split(x).reshape(B, T, self.speakers, C)
        return self.split_norm(x).transpose(1, 2)  # [B, J, T, F]

    def forward(self, x):
        x, skips = self.encoder(x)

        # The same Speaker Split processes the bottleneck and every skip
        x = self.split_speakers(x)
        auxiliary_features = []

        for stage, skip in zip(self.decoder, reversed(skips)):
            x = stage(x, self.split_speakers(skip))
            if self.training:
                auxiliary_features.append(x)

        return self.output(x), auxiliary_features


class SepReformer(nn.Module):
    def __init__(self, dropout=0.1, layer_scale_init=0.01):
        super().__init__()
        # [see] 4.2 -> SepReformer-T configuration
        audio_dim, dim, heads = 256, 64, 8
        levels, speakers, radius = 4, 2, 32
        self.kernel_size, self.stride = 16, 4
        self.frame_multiple = 2 ** levels

        self.audio_encoder = AudioEncoder(1, audio_dim, self.kernel_size, self.stride)
        self.separator = Separator(audio_dim, dim, heads, levels, speakers, radius, dropout, layer_scale_init)
        self.audio_decoder = nn.ConvTranspose1d(audio_dim, 1, self.kernel_size, stride=self.stride)

        # [see] 3.4 -> one auxiliary mask head and audio decoder per level
        self.aux_heads = nn.ModuleList(nn.Sequential(nn.Linear(dim, 2 * dim), nn.GLU(), nn.Linear(dim, audio_dim)) for _ in range(levels))
        self.aux_decoders = nn.ModuleList(nn.ConvTranspose1d(audio_dim, 1, self.kernel_size, stride=self.stride) for _ in range(levels))

    @staticmethod
    def decode(x, decoder, length):
        B, J, T, C = x.shape
        x = x.reshape(B * J, T, C).transpose(1, 2)
        return decoder(x).reshape(B, J, -1)[..., :length]

    def forward(self, mix, **batch):
        # Pad to a multiple of 2**R encoded frames so every upsample matches its skip
        mix = mix.unsqueeze(1)
        remaining = max(0, mix.size(-1) - self.kernel_size)
        frames = 1 + (remaining + self.stride - 1) // self.stride
        frames += (-frames) % self.frame_multiple
        padded_length = self.kernel_size + (frames - 1) * self.stride
        mixture = self.audio_encoder(F.pad(mix, (0, padded_length - mix.size(-1))))
        final, features = self.separator(mixture)

        # Decode the main speaker estimates directly and crop back to N samples.
        output = {"logits": self.decode(final, self.audio_decoder, mix.size(-1))}

        if features:
            mixture = mixture.transpose(1, 2).unsqueeze(1)  # [B, 1, T, F0]
            T = mixture.size(2)
            auxiliary = []
            for x, head, decoder in zip(features, self.aux_heads, self.aux_decoders):
                # Upsample the stage mask and apply it to the full-resolution mixture.
                factor = T // x.size(2)
                mask = head(x).repeat_interleave(factor, dim=2)
                auxiliary.append(self.decode(mask * mixture, decoder, mix.size(-1)))
            output["aux_logits"] = auxiliary

        return output
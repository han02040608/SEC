from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn


class EnergyAmplificationBlock(nn.Module):
    """Amplify low-energy frequency components."""

    def __init__(self, hidden_dim: int, amplify_ratio: float = 2.0) -> None:
        super().__init__()
        self.amplify_ratio = amplify_ratio
        self.energy_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.threshold_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        seq_len = x.size(1)
        x_freq = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        energy = amplitude.mean(dim=1, keepdim=True)
        threshold = self.threshold_net(energy)
        low_energy_mask = (amplitude < threshold).float()

        importance = self.energy_analyzer(amplitude)
        amplification = 1.0 + (self.amplify_ratio - 1.0) * importance * low_energy_mask

        amplified_freq = amplitude * amplification * torch.exp(1j * phase)
        amplified_x = torch.fft.irfft(amplified_freq, n=seq_len, dim=1)

        return amplified_x, {"amplification": amplification}


class EnergyRestorationBlock(nn.Module):
    """Restore spectral magnitude after amplification."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.recovery_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, energy_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        seq_len = x.size(1)
        x_freq = torch.fft.rfft(x, dim=1)
        amplitude = torch.abs(x_freq)
        phase = torch.angle(x_freq)

        amplification = energy_info["amplification"]
        recovery_ratio = 1.0 / (amplification + 1e-8)
        recovered_freq = amplitude * recovery_ratio * torch.exp(1j * phase)
        recovered_x = torch.fft.irfft(recovered_freq, n=seq_len, dim=1)

        return recovered_x + self.recovery_net(recovered_x)


class FrequencyDomainFeatureDecouplingModule(nn.Module):
    """Decouple hidden states into seasonal and trend components."""

    def __init__(self, hidden_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        self.seasonal_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.trend_pool = nn.AvgPool1d(
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )
        self.seasonal_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.trend_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        seasonal = self.seasonal_conv(x_t).transpose(1, 2)
        trend = self.trend_pool(x_t).transpose(1, 2)
        combined = torch.cat(
            [self.seasonal_predictor(seasonal), self.trend_predictor(trend)],
            dim=-1,
        )
        return self.fusion(combined)


class VariableInteractionCommonalityModule(nn.Module):
    """Model commonality and specificity across variables."""

    def __init__(self, hidden_dim: int, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(hidden_dim, hidden_dim // 4, 1),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 4, hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.channel_specific = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, 1, groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.temporal_enhancement = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        channel_common = (x_t * self.channel_attention(x_t)).transpose(1, 2)
        channel_specific = self.channel_specific(x_t).transpose(1, 2)
        temporal_enhanced, _ = self.temporal_enhancement(
            channel_common, channel_common, channel_common
        )
        return self.fusion(torch.cat([temporal_enhanced, channel_specific], dim=-1))


class TemporalGraphLayer(nn.Module):
    """Temporal message passing within a local window."""

    def __init__(self, hidden_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.message_net = nn.Linear(hidden_dim, hidden_dim)
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.edge_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        updated_nodes = []

        for index in range(seq_len):
            start = max(0, index - self.kernel_size // 2)
            end = min(seq_len, index + self.kernel_size // 2 + 1)

            current_node = x[:, index : index + 1, :]
            neighbors = x[:, start:end, :]
            current_expanded = current_node.expand(-1, neighbors.size(1), -1)

            edge_features = torch.cat([current_expanded, neighbors], dim=-1)
            edge_weights = self.edge_weight_net(edge_features)
            messages = self.message_net(neighbors)
            aggregated = (messages * edge_weights).sum(dim=1)

            updated_node = self.update_net(
                torch.cat([current_node.squeeze(1), aggregated], dim=-1)
            )
            updated_nodes.append(updated_node.unsqueeze(1))

        return torch.cat(updated_nodes, dim=1)


class MultiScaleTemporalGraphNeuralNetwork(nn.Module):
    """Aggregate temporal graph features across multiple receptive fields."""

    def __init__(self, hidden_dim: int, kernel_sizes: Sequence[int] = (3, 5, 7)) -> None:
        super().__init__()
        self.graph_layers = nn.ModuleList(
            [TemporalGraphLayer(hidden_dim, kernel_size) for kernel_size in kernel_sizes]
        )
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multi_scale_features = [layer(x) for layer in self.graph_layers]
        return self.scale_fusion(torch.cat(multi_scale_features, dim=-1))


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        ff_dim = ff_dim or hidden_dim * 4
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_input = self.norm1(x)
        attention_output, attention_weights = self.self_attention(
            attention_input, attention_input, attention_input, attn_mask=attn_mask
        )
        x = x + self.dropout(attention_output)
        x = x + self.feed_forward(self.norm2(x))
        return x, attention_weights


class GlobalDependencyEncoder(nn.Module):
    """Capture long-range temporal dependencies with transformer layers."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(hidden_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attention_weights: Optional[torch.Tensor] = None
        for layer in self.layers:
            x, attention_weights = layer(x)
        return x, attention_weights


class SpectralEnergyModulator(nn.Module):
    """Spectral Energy Modulator composed of amplification and restoration blocks."""

    def __init__(self, hidden_dim: int, amplify_ratio: float = 2.0) -> None:
        super().__init__()
        self.energy_amplification_block = EnergyAmplificationBlock(hidden_dim, amplify_ratio)
        self.energy_restoration_block = EnergyRestorationBlock(hidden_dim)

    def amplify(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.energy_amplification_block(x)

    def restore(self, x: torch.Tensor, energy_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.energy_restoration_block(x, energy_info)


class SpectralEnhancedCrossDomainNetwork(nn.Module):
    """Spectral-Enhanced Cross-domain Network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_gnn_layers: int = 2,
        num_transformer_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        kernel_sizes: Sequence[int] = (3, 5, 7),
        amplify_ratio: float = 2.0,
        max_seq_length: int = 1000,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.position_embedding = nn.Parameter(
            torch.randn(1, max_seq_length, hidden_dim) * 0.02
        )
        self.spectral_energy_modulator = SpectralEnergyModulator(hidden_dim, amplify_ratio)
        self.frequency_domain_feature_decoupling_module = (
            FrequencyDomainFeatureDecouplingModule(hidden_dim)
        )
        self.variable_interaction_commonality_module = (
            VariableInteractionCommonalityModule(hidden_dim, input_dim)
        )
        self.multi_scale_temporal_graph_neural_network_layers = nn.ModuleList(
            [
                MultiScaleTemporalGraphNeuralNetwork(hidden_dim, kernel_sizes)
                for _ in range(num_gnn_layers)
            ]
        )
        self.global_dependency_encoder = GlobalDependencyEncoder(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len = x.size(1)
        if seq_len > self.position_embedding.size(1):
            raise ValueError(
                f"Input sequence length {seq_len} exceeds max_seq_length "
                f"{self.position_embedding.size(1)}."
            )

        x = self.input_projection(x)
        x = x + self.position_embedding[:, :seq_len, :]
        x, energy_info = self.spectral_energy_modulator.amplify(x)
        x = self.frequency_domain_feature_decoupling_module(x)
        x = self.variable_interaction_commonality_module(x)

        for gnn_layer in self.multi_scale_temporal_graph_neural_network_layers:
            x = x + gnn_layer(x)

        x, attention_weights = self.global_dependency_encoder(x)
        x = self.spectral_energy_modulator.restore(x, energy_info)
        output = self.output_head(x[:, -1, :])
        return output, attention_weights


class SECModel(SpectralEnhancedCrossDomainNetwork):
    """Alias for the full SEC model."""

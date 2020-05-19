from argparse import ArgumentParser, Namespace
import math

import pytorch_lightning as pl
import torch
from torch import nn

from typing import Dict, Tuple, Type, Any, Optional, Union


class LSTM(pl.LightningModule):
    r"""
    An LSTM base

    hparams
    --------
    The default values for these parameters are set in add_base_specific_args

    :params hparams.num_layers: The number of LSTM layers to use. Default = 1
    :params hparams.dropout: The LSTM dropout to use. If hparams.num_layers ==1, this
        dropout is applied between timesteps. Otherwise, it is applied between layers.
        Default = 0
    """

    def __init__(self, input_size: int, hparams: Namespace) -> None:
        super().__init__()

        self.hparmas = hparams

        if (hparams.num_lstm_layers > 1) or (hparams.lstm_dropout == 0):
            # if we can, use the default LSTM implementation
            self.lstm: Union[nn.LSTM, UnrolledLSTM] = nn.LSTM(
                input_size=input_size,
                hidden_size=hparams.hidden_vector_size,
                dropout=hparams.lstm_dropout,
                batch_first=True,
                num_layers=hparams.num_lstm_layers,
            )
        else:
            # if the LSTM is only one layer,
            # we will apply dropout between timesteps
            # instead of between layers
            self.lstm = UnrolledLSTM(
                input_size=input_size,
                hidden_size=hparams.hidden_vector_size,
                dropout=hparams.lstm_dropout,
                batch_first=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, (hn, cn) = self.lstm(x)
        return hn[-1, :, :]

    @staticmethod
    def add_base_specific_arguments(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser_args: Dict[str, Tuple[Type, Any]] = {
            "--num_lstm_layers": (int, 1),
            "--lstm_dropout": (float, 0.2),
        }

        for key, vals in parser_args.items():
            parser.add_argument(key, type=vals[0], default=vals[1])

        return parser


class UnrolledLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float, batch_first: bool
    ) -> None:
        super().__init__()

        self.batch_first = batch_first
        self.hidden_size = hidden_size

        self.rnn = UnrolledLSTMCell(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.dropout = nn.Dropout(dropout)
        self.initialize_weights()

    def initialize_weights(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.rnn.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

    def forward(  # type: ignore
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        sequence_length = x.shape[1] if self.batch_first else x.shape[0]
        batch_size = x.shape[0] if self.batch_first else x.shape[1]

        if state is None:
            # initialize to zeros
            hidden, cell = (
                torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size),
            )

            if x.is_cuda:
                hidden, cell = hidden.cuda(), cell.cuda()
        else:
            hidden, cell = state

        outputs = []
        for i in range(sequence_length):
            input_x = x[:, i, :].unsqueeze(1)
            _, (hidden, cell) = self.rnn(input_x, (hidden, cell))
            outputs.append(hidden)
            hidden = self.dropout(hidden)

        return torch.stack(outputs, dim=0), (hidden, cell)


class UnrolledLSTMCell(nn.Module):
    """An unrolled LSTM, so that dropout can be applied between
    timesteps instead of between layers
    """

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.forget_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size,
                    out_features=hidden_size,
                    bias=True,
                ),
                nn.Sigmoid(),
            ]
        )

        self.update_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size,
                    out_features=hidden_size,
                    bias=True,
                ),
                nn.Sigmoid(),
            ]
        )

        self.update_candidates = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size,
                    out_features=hidden_size,
                    bias=True,
                ),
                nn.Tanh(),
            ]
        )

        self.output_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size,
                    out_features=hidden_size,
                    bias=True,
                ),
                nn.Sigmoid(),
            ]
        )

        self.cell_state_activation = nn.Tanh()

    def forward(  # type: ignore
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden, cell = state

        if self.batch_first:
            hidden, cell = torch.transpose(hidden, 0, 1), torch.transpose(cell, 0, 1)

        forget_state = self.forget_gate(torch.cat((x, hidden), dim=-1))
        update_state = self.update_gate(torch.cat((x, hidden), dim=-1))
        cell_candidates = self.update_candidates(torch.cat((x, hidden), dim=-1))

        updated_cell = (forget_state * cell) + (update_state * cell_candidates)

        output_state = self.output_gate(torch.cat((x, hidden), dim=-1))
        updated_hidden = output_state * self.cell_state_activation(updated_cell)

        if self.batch_first:
            updated_hidden = torch.transpose(updated_hidden, 0, 1)
            updated_cell = torch.transpose(updated_cell, 0, 1)

        return updated_hidden, (updated_hidden, updated_cell)

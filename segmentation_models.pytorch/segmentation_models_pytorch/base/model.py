import torch

from . import initialization as init
from .hub_mixin import SMPHubMixin


class SegmentationModel(torch.nn.Module, SMPHubMixin):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


class SegmentationModelDecoupledDecoders(torch.nn.Module, SMPHubMixin):
    def initialize(self):
        init.initialize_decoder(self.decoder_semantic)
        init.initialize_decoder(self.decoder_centers)
        init.initialize_decoder(self.decoder_offsets)

        init.initialize_head(self.head_semantic)
        init.initialize_head(self.head_centers)
        init.initialize_head(self.head_offsets)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        decoder_semantic_output = self.decoder_semantic(*features)
        decoder_centers_output = self.decoder_centers(*features)
        decoder_offsets_output = self.decoder_offsets(*features)

        masks_semantic = self.head_semantic(decoder_semantic_output)
        masks_centers = self.head_centers(decoder_centers_output)
        masks_offsets = self.head_offsets(decoder_offsets_output)

        assert self.classification_head is None

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return {
            'masks_semantic': masks_semantic,
            'masks_centers': masks_centers,
            'masks_offsets': masks_offsets,
        }

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x


class SegmentationModelDecoupledDecodersBoundaries(torch.nn.Module, SMPHubMixin):
    def initialize(self):
        init.initialize_decoder(self.decoder_semantic)
        init.initialize_decoder(self.decoder_centers)
        init.initialize_decoder(self.decoder_offsets)

        init.initialize_head(self.head_semantic)
        init.initialize_head(self.head_boundaries)
        init.initialize_head(self.head_centers)
        init.initialize_head(self.head_offsets)

        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)

        decoder_semantic_output = self.decoder_semantic(*features)
        decoder_centers_output = self.decoder_centers(*features)
        decoder_offsets_output = self.decoder_offsets(*features)

        masks_semantic = self.head_semantic(decoder_semantic_output)
        masks_boundaries = self.head_boundaries(decoder_semantic_output)
        masks_centers = self.head_centers(decoder_centers_output)
        masks_offsets = self.head_offsets(decoder_offsets_output)

        assert self.classification_head is None

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return {
            'masks_semantic': masks_semantic,
            'masks_boundaries': masks_boundaries,
            'masks_centers': masks_centers,
            'masks_offsets': masks_offsets,
        }

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x



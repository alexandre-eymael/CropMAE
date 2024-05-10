from .components import MAEViT, DecoderBlock
import torch.nn as nn
import torch
from functools import partial

class SiameseMAE(MAEViT):
    
    @property
    def n_params(self, unit=1e6, ndigits=4):
        """ Number of parameters in model, divided by `unit`, rounded to `ndigits` digits """
        count_params = lambda params: round(sum(params) / unit, ndigits)
        return {
            "total_n_params" : count_params(p.numel() for p in self.parameters()),
            "total_n_trainable_params" : count_params(p.numel() for p in self.parameters() if p.requires_grad),
            "total_n_trainable_params_encoder" : count_params(p.numel() for p in self.blocks.parameters()),
            "total_n_trainable_params_decoder" : count_params(p.numel() for p in self.decoder_blocks.parameters()),
        }

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Overwrite decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(kwargs["decoder_embed_dim"], kwargs["decoder_num_heads"])
            for _ in range(kwargs["decoder_depth"])
        ])

        print(f"\nSiamMAE with: Decoder Embed Dim: {kwargs['decoder_embed_dim']}, Decoder Depth: {kwargs['decoder_depth']}, Decoder Num Heads: {kwargs['decoder_num_heads']}")

        # Reinitialize weights
        self.initialize_weights()

    def forward_encoder(self, base_x, mask_ratio):
        # embed patches
        x = self.patch_embed(base_x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_no_masking(self, x):

        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    
    # Utility functions for the decoder
    def _append_mask_token(self, x_latent, x_ids_restore):
        x_mask_tokens = self.mask_token.repeat(x_latent.shape[0], x_ids_restore.shape[1] + 1 - x_latent.shape[1], 1)            # [2, 147, 512]
        x_ = torch.cat([x_latent[:, 1:, :], x_mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=x_ids_restore.unsqueeze(-1).repeat(1, 1, x_latent.shape[2]))  # unshuffle
        x = torch.cat([x_latent[:, :1, :], x_], dim=1)  # append cls token      
        return x  
    
    def _apply_decoder_blocks(self, q, kv):
        for blk in self.decoder_blocks:
            q = blk(q, kv)
        return q

    def forward_decoder(self, x_unmasked, x_masked_latent, x_masked_ids_restore):

        # embed tokens
        x_unmasked = self.decoder_embed(x_unmasked)
        x_masked_latent = self.decoder_embed(x_masked_latent)

        # append mask token
        x_masked_latent = self._append_mask_token(x_masked_latent, x_masked_ids_restore)

        # Add pos embed
        x_masked_latent = x_masked_latent + self.decoder_pos_embed

        # Apply decoder blocks
        x_masked_latent = self._apply_decoder_blocks(x_masked_latent, x_unmasked)

        # Normalize
        x_masked_latent = self.decoder_norm(x_masked_latent)

        # Predictor projection
        x_masked_latent = self.decoder_pred(x_masked_latent)

        # Remove cls token
        x_masked_latent = x_masked_latent[:, 1:, :] 

        return x_masked_latent
    

    def forward(self, imgs, mask_ratio):

        assert imgs.shape[1] == 2, "Number of frames must be equal to 2"

        # Split imgs into unmasked and masked frames
        # The unmasked frame is the first frame, while all the other frames are masked
        imgs = imgs.chunk(2, dim=1)
        imgs = [img.squeeze(1) for img in imgs]
        unmasked_img, masked_img = imgs[0], imgs[1]

        # Encode 
        unmasked_img = self.forward_encoder_no_masking(unmasked_img)
        masked_latent, masked_mask, masked_ids_restore = self.forward_encoder(masked_img, mask_ratio)

        # Decode
        masked_pred = self.forward_decoder(unmasked_img, masked_latent, masked_ids_restore)

        # Compute loss
        loss = self.forward_loss(masked_img, masked_pred, masked_mask)

        return loss, [masked_pred], [masked_mask]
    

def siam_mae_vit_small(patch_size=16, decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def siam_mae_vit_base(patch_size=16, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def siam_mae_vit_large(patch_size=16, decoder_embed_dim=1024, decoder_depth=12, decoder_num_heads=16, **kwargs):
    model = SiameseMAE(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
    
SIAM_MODELS = {
    "vits" : siam_mae_vit_small,
    "vitb" : siam_mae_vit_base,
    "vitl" : siam_mae_vit_large
}
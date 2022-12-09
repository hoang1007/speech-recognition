from easydict import EasyDict as dict

D_MODEL = 768
HIDDEN_SIZE = 512



context_encoder = dict(
    feature_projection=dict(
        in_features=HIDDEN_SIZE,
        out_features=D_MODEL,
        dropout=0.1,
    ),
    encoder=dict(
        d_model=D_MODEL,
        num_layers=12,
        layer_drop=0.05,
        pos_embedding=dict(
            d_model=D_MODEL,
            kernel_size=3,
            groups=2,
            dropout=0.1,
        ),
        layer=dict(
            d_model=D_MODEL,
            num_heads=8,
            layer_norm_first=False,
            feed_forward_dim=2048,
            dropout=0.1,
        ),
    )
)

feature_extractor = dict(
    num_channels=7 * (HIDDEN_SIZE,),
    kernel_sizes=(10,) + 4 * (3,) + 2 * (2,),
    strides=(5,) + 6 * (2,),
)

quantizer = dict(
    in_features=HIDDEN_SIZE,
    num_codebooks=2,
    num_codewords=320,
    d_model=D_MODEL,
)

wav2vec2_pretraining = dict(
    context_encoder=context_encoder,
    feature_extractor=feature_extractor,
    quantizer=quantizer,
    mask_prob=0.65,
    mask_length=10,
    min_masks=2,
    num_negatives=100,
    contrastive_logits_temperature=0.1,
    diversity_loss_weight=0.2,
)
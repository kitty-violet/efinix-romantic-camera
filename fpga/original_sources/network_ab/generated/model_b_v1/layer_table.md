# Derived from model_b_contract.json; do not edit.

Contract SHA256: `b330b3bce8d09544d596ff446a457c60c77407ded12800cd9cceb16fba92abf3`

|id|kind|input HWC|output HWC|BN|train bias|activation|quantization|
|---|---|---|---|---|---|---|---|
|stem|conv3x3|[720, 1280, 3]|[720, 1280, 8]|True|False|relu|output_int8|
|down1.dw|dw3x3|[720, 1280, 8]|[360, 640, 8]|True|False|relu|output_int8|
|down1.pw|pw1x1|[360, 640, 8]|[360, 640, 16]|True|False|relu|output_int8|
|down2.dw|dw3x3|[360, 640, 16]|[180, 320, 16]|True|False|relu|output_int8|
|down2.pw|pw1x1|[180, 320, 16]|[180, 320, 32]|True|False|relu|output_int8|
|res1_conv1.dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res1_conv1.pw|pw1x1|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res1_conv2.dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res1_conv2.pw|pw1x1|[180, 320, 32]|[180, 320, 32]|True|False|none|output_int8|
|res1_add|residual_add|[180, 320, 32]|[180, 320, 32]|False|False|none|residual_int8|
|res2_conv1.dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res2_conv1.pw|pw1x1|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res2_conv2.dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res2_conv2.pw|pw1x1|[180, 320, 32]|[180, 320, 32]|True|False|none|output_int8|
|res2_add|residual_add|[180, 320, 32]|[180, 320, 32]|False|False|none|residual_int8|
|res3_conv1.dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res3_conv1.pw|pw1x1|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res3_conv2.dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|res3_conv2.pw|pw1x1|[180, 320, 32]|[180, 320, 32]|True|False|none|output_int8|
|res3_add|residual_add|[180, 320, 32]|[180, 320, 32]|False|False|none|residual_int8|
|up1_low_dw|dw3x3|[180, 320, 32]|[180, 320, 32]|True|False|relu|output_int8|
|up1_low_pw|pw1x1|[180, 320, 32]|[180, 320, 16]|True|False|relu|output_int8|
|up1_nn|nearest_x2|[180, 320, 16]|[360, 640, 16]|False|False|none|inherit_scale|
|up1_high_dw|dw3x3|[360, 640, 16]|[360, 640, 16]|True|False|relu|output_int8|
|up2_low_dw|dw3x3|[360, 640, 16]|[360, 640, 16]|True|False|relu|output_int8|
|up2_low_pw|pw1x1|[360, 640, 16]|[360, 640, 8]|True|False|relu|output_int8|
|up2_nn|nearest_x2|[360, 640, 8]|[720, 1280, 8]|False|False|none|inherit_scale|
|up2_high_dw|dw3x3|[720, 1280, 8]|[720, 1280, 8]|True|False|relu|output_int8|
|rgb_head|pw1x1|[720, 1280, 8]|[720, 1280, 3]|False|True|clamp_rgb|output_int8|

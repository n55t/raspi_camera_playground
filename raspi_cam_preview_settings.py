from picamera2 import Picamera2
from libcamera import controls
import cv2

# ===== 設定パラメータ =====

# --- ① 露光・AE 系 ---
EXPOSURE_TIME_US = 100000
ANALOGUE_GAIN    = 4.0
AE_ENABLE        = False
EV_COMP          = 0.0
# AE_METERING_MODE    = controls.AeMeteringModeEnum.CentreWeighted
# AE_CONSTRAINT_MODE  = controls.AeConstraintModeEnum.Normal

# --- ② ホワイトバランス・色 ---
AWB_ENABLE   = False
COLOUR_GAINS = (1.4, 1.2)
SATURATION   = 1.0

# --- ③ 明るさ・コントラスト・画質 ---
BRIGHTNESS = 0.0
CONTRAST   = 1.2
SHARPNESS  = 0.0

# --- ④ フレーム・解像度・センサ領域 ---
FRAME_DURATION_LIMITS = (33333, 33333)
SCALER_CROP = None

# --- ⑤ ノイズ・画質最適化 ---
NOISE_REDUCTION_MODE = controls.draft.NoiseReductionModeEnum.Off
# TONEMAP_MODE = controls.draft.TonemapModeEnum.Off

# --- ⑥ フォーカス・レンズ ---
# AF_MODE       = controls.AfModeEnum.Manual
# LENS_POSITION = 2.0

# --- ⑦ HDR・ダイナミックレンジ ---
# HDR_MODE = controls.draft.HdrModeEnum.Off

# --- ⑧ その他センサ／メタ情報 ---
# SENSOR_TEMPERATURE = None
# SENSOR_TIMESTAMP   = None

# --- ⑨ その他コントロール ---
DISPLAY_MAX_WIDTH  = 1280   # 画面表示サイズ幅
DISPLAY_MAX_HEIGHT = 720    # 画面表示サイズ高さ

# ============================

picam2 = Picamera2()

# センサの最大解像度を取得
sensor_width, sensor_height = picam2.sensor_resolution

config = picam2.create_preview_configuration(
    main={"size": (sensor_width, sensor_height)}
)


picam2.configure(config)
picam2.start()

ctrls = {
    "ExposureTime": EXPOSURE_TIME_US,
    "AnalogueGain": ANALOGUE_GAIN,
    "AeEnable": AE_ENABLE,
    "ExposureValue": EV_COMP,

    "AwbEnable": AWB_ENABLE,
    "ColourGains": COLOUR_GAINS,
    "Saturation": SATURATION,

    "Brightness": BRIGHTNESS,
    "Contrast": CONTRAST,
    "Sharpness": SHARPNESS,

    "FrameDurationLimits": FRAME_DURATION_LIMITS,
    "NoiseReductionMode": NOISE_REDUCTION_MODE,
}

if SCALER_CROP is not None:
    ctrls["ScalerCrop"] = SCALER_CROP

picam2.set_controls(ctrls)

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    h, w = frame.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cv2.imshow("cam", display_frame)
    if cv2.waitKey(1) == 27:
        break

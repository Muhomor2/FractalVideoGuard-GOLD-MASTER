# FractalVideoGuard v0.5.2 - Configuration Guide
## Система конфигурируемых параметров для пользователей

---

## 🎯 БЫСТРЫЙ СТАРТ

### 1. Использование дефолтных параметров

```python
from fio_features import extract_features
from fio_config import FIOConfig

# Автоматически использует оптимальные параметры
config = FIOConfig()
features, debug, E, D = extract_features('video.mp4', config=config)
```

### 2. Выбор готового пресета

```python
from fio_config import ConfigPresets

# Production High Quality - максимальное качество
config = ConfigPresets.production_high_quality()

# Production Fast - баланс скорость/качество для real-time
config = ConfigPresets.production_fast()

# Research Debug - максимум детализации для исследований
config = ConfigPresets.research_debug()

# Mobile Lightweight - минимальные ресурсы для edge/mobile
config = ConfigPresets.mobile_lightweight()

features, debug, E, D = extract_features('video.mp4', config=config)
```

### 3. Кастомизация отдельных параметров

```python
config = FIOConfig()

# Изменить параметры видео
config.video.fps_target = 15        # Увеличить FPS выборки
config.video.max_frames = 1200      # Обрабатывать больше кадров

# Изменить ROI extraction
config.roi.min_roi_side = 64        # Минимальный размер лица
config.roi.std_roi_side = 512       # Стандартный размер для DCT/FFT

# Изменить fractal features
config.fractal.dfa_min_rsquared = 0.95  # Более строгая проверка качества

# Изменить frequency features
config.frequency.sample_rate_frames = 3  # Чаще сэмплировать кадры

# Проверить корректность
errors = config.validate()
if not config.is_valid():
    print("Ошибки конфигурации:", errors)
else:
    print("✅ Конфигурация валидна")
```

---

## 📋 ПОДРОБНОЕ ОПИСАНИЕ ПАРАМЕТРОВ

### Video Processing Parameters

```python
config.video.fps_target = 12              # Целевой FPS для extraction
config.video.max_frames = 900             # Макс. кадров (≈75 сек @ 12fps)
config.video.rotation_timeout_sec = 2.0   # Таймаут для rotation metadata
config.video.rotation_fallback_enable = True  # Fallback rotation
config.video.min_resolution = (320, 240)  # Мин. разрешение видео
config.video.max_resolution = (4096, 2160)  # Макс. разрешение (4K)
```

**Когда менять:**
- `fps_target` ↑ → больше temporal detail, медленнее
- `max_frames` ↑ → обрабатывать длиннее видео
- `rotation_timeout_sec` ↑ → для медленных storage/network

---

### ROI (Region of Interest) Parameters

```python
config.roi.use_mediapipe = True           # Использовать MediaPipe face detection
config.roi.detection_confidence = 0.5     # Мин. уверенность детекции лица
config.roi.min_roi_side = 48              # Мин. размер ROI (px)
config.roi.std_roi_side = 256             # Стандартный размер (power of 2)
config.roi.max_roi_side = 512             # Макс. размер (защита памяти)
config.roi.blur_threshold = 100.0         # Порог Laplacian variance для blur
config.roi.brightness_range = (20, 235)   # Допустимая яркость
config.roi.bbox_smoothing_alpha = 0.65    # EMA сглаживание bbox
config.roi.center_crop_fraction = 0.62    # Fallback center crop
```

**Когда менять:**
- `min_roi_side` ↓ → детектировать более мелкие лица (дальний план)
- `std_roi_side` ↑ → больше деталей для DCT/FFT, больше памяти/CPU
- `blur_threshold` ↑ → отбрасывать больше размытых кадров
- `bbox_smoothing_alpha` → 0.0 (responsive) vs 0.9 (stable)

---

### Fractal Features Parameters

```python
config.fractal.dfa_scales = (8, 16, 32, 64, 128, 256, 512)
config.fractal.dfa_min_rsquared = 0.90    # Качество DFA фита
config.fractal.dfa_poly_order = 1         # Detrending: 1=linear, 2=quadratic
config.fractal.boxcount_scales = (2, 4, 8, 16, 32, 64)
config.fractal.boxcount_min_rsquared = 0.85
config.fractal.canny_threshold1 = 80      # Canny edge detection
config.fractal.canny_threshold2 = 160
config.fractal.highpass_sigma = 1.2       # Gaussian highpass filter

# Universal Attractor Theory (QO3/FIO)
config.fractal.theoretical_h_real = 0.70  # Expected H for real videos
config.fractal.theoretical_h_fake = 0.55  # Expected H for GAN videos
config.fractal.theoretical_d_real = 1.35  # Expected D for real edges
config.fractal.theoretical_d_fake = 1.15  # Expected D for synthetic edges
```

**Когда менять:**
- `dfa_min_rsquared` ↑ → более строгая валидация (отбросить зашумленные)
- `dfa_poly_order` = 2 → для видео с сильными трендами
- `theoretical_*` → адаптировать под конкретные GAN модели

---

### Frequency Artifact Parameters

```python
config.frequency.dct_block_size = 8       # DCT block (8x8 для JPEG/H.264)
config.frequency.dct_hf_threshold = 5     # High-freq DCT threshold
config.frequency.fft_hf_ratio = 0.25      # FFT high-freq band (outer 25%)
config.frequency.blockiness_grid_size = 8 # JPEG block grid
config.frequency.ringing_median_ksize = 3 # Median blur для ringing
config.frequency.ringing_laplacian_ksize = 3
config.frequency.ringing_epsilon_relative = 0.01  # MAD epsilon (1% range)
config.frequency.block_var_size = 8
config.frequency.sample_rate_frames = 6   # Обработать каждый Nth frame
config.frequency.nan_handling = 'omit'    # 'omit', 'zero', 'mean'
```

**Когда менять:**
- `sample_rate_frames` ↓ → больше precision, медленнее
- `nan_handling` → 'mean' для консистентности feature векторов
- `ringing_epsilon_relative` ↑ → меньше чувствительность к низкоконтрастным видео

---

### Statistical Analysis Parameters

```python
config.statistics.enable_bootstrap_ci = True
config.statistics.bootstrap_n_samples = 250
config.statistics.bootstrap_confidence = 0.95
config.statistics.bootstrap_min_data = 80
config.statistics.enable_surrogate_test = True
config.statistics.surrogate_n_samples = 120
config.statistics.random_seed = 2026      # Воспроизводимость
```

**Когда менять:**
- `bootstrap_n_samples` ↑ → более точные CI, медленнее
- `enable_surrogate_test = False` → ускорить production inference
- `random_seed` → фиксировать для A/B тестов

---

### Training Parameters

```python
config.training.model_type = 'logistic'   # 'logistic', 'randomforest', 'xgboost'
config.training.enable_calibration = True
config.training.calibration_method = 'isotonic'  # 'isotonic', 'sigmoid'
config.training.cv_folds = 5
config.training.cv_stratify = True
config.training.min_feature_variance = 0.01
config.training.max_feature_correlation = 0.95
config.training.l2_penalty = 1.0          # Logistic C parameter
config.training.balance_classes = True
```

**Когда менять:**
- `model_type = 'xgboost'` → для больших датасетов
- `cv_folds` ↑ → более robustная оценка, медленнее
- `l2_penalty` ↓ → сильнее regularization (при overfitting)

---

## 💾 СОХРАНЕНИЕ И ЗАГРУЗКА КОНФИГУРАЦИЙ

### JSON формат

```python
# Экспорт в JSON
config = FIOConfig()
config.video.fps_target = 15
config.to_json(Path('my_config.json'))

# Загрузка из JSON
config_loaded = FIOConfig.from_json(Path('my_config.json'))
```

### Environment Variables

```bash
# Устанавливать через переменные окружения
export FIO_VIDEO_FPS_TARGET=15
export FIO_VIDEO_MAX_FRAMES=1200
export FIO_ROI_MIN_ROI_SIDE=64
export FIO_FRACTAL_DFA_MIN_RSQUARED=0.95
export FIO_FREQUENCY_SAMPLE_RATE_FRAMES=3

# Затем в коде:
config = FIOConfig.from_env(prefix='FIO_')
```

### Python Dictionary

```python
custom_params = {
    'video': {
        'fps_target': 15,
        'max_frames': 1200,
    },
    'roi': {
        'min_roi_side': 64,
        'std_roi_side': 512,
    },
    'fractal': {
        'dfa_min_rsquared': 0.95,
    }
}

config = FIOConfig.from_dict(custom_params)
```

---

## 🔧 CLI ИНСТРУМЕНТЫ

### 1. Валидация конфигурации

```bash
python fio_config.py --validate my_config.json

# Output:
# ✅ Configuration is valid!
# или
# ❌ Configuration validation FAILED:
# [video]
#   - fps_target=0 out of range [1, 120]
```

### 2. Экспорт дефолтной конфигурации

```bash
python fio_config.py --export default_config.json
# ✅ Exported default configuration to: default_config.json
```

### 3. Показать дефолты

```bash
python fio_config.py --show-defaults
# Выводит JSON со всеми дефолтными параметрами
```

### 4. Использовать пресет

```bash
python fio_config.py --preset high_quality
python fio_config.py --preset fast
python fio_config.py --preset debug
python fio_config.py --preset mobile
```

---

## 🎬 ПРИМЕРЫ USE CASES

### Use Case 1: High-Quality Forensic Analysis

```python
from fio_config import ConfigPresets

# Максимальное качество для форензики
config = ConfigPresets.production_high_quality()

# Дополнительная кастомизация
config.video.fps_target = 20              # Еще больше temporal resolution
config.roi.std_roi_side = 512             # Максимальный ROI
config.statistics.bootstrap_n_samples = 1000  # Точнее CI

features = extract_features('evidence.mp4', config=config)
```

### Use Case 2: Real-Time Detection System

```python
config = ConfigPresets.production_fast()

# Оптимизация для скорости
config.video.max_frames = 300             # Быстрее на коротких клипах
config.frequency.sample_rate_frames = 15  # Реже сэмплировать
config.statistics.enable_bootstrap_ci = False  # Отключить CI
config.statistics.enable_surrogate_test = False  # Отключить surrogates

# Батч-процессинг с параллелизмом
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(extract_features, video, config=config)
        for video in video_list
    ]
```

### Use Case 3: Mobile/Edge Deployment

```python
config = ConfigPresets.mobile_lightweight()

# Еще более aggressive optimization
config.video.fps_target = 4
config.video.max_frames = 200
config.roi.use_mediapipe = False  # CPU-only fallback
config.roi.std_roi_side = 128     # Меньше память
config.frequency.sample_rate_frames = 20

features = extract_features('mobile_video.mp4', config=config)
```

### Use Case 4: Research - Параметрический sweep

```python
config = FIOConfig()

# Grid search по ключевым параметрам
results = []

for fps in [8, 12, 16, 20]:
    for roi_size in [128, 256, 512]:
        for sample_rate in [3, 6, 10]:
            config.video.fps_target = fps
            config.roi.std_roi_side = roi_size
            config.frequency.sample_rate_frames = sample_rate
            
            features = extract_features('test.mp4', config=config)
            results.append({
                'fps': fps,
                'roi_size': roi_size,
                'sample_rate': sample_rate,
                'h_dfa': features.h_dfa,
                'd_mean': features.d_mean,
            })

# Анализ результатов
import pandas as pd
df = pd.DataFrame(results)
print(df.groupby(['fps', 'roi_size']).mean())
```

---

## ⚠️ ВАЖНЫЕ РЕКОМЕНДАЦИИ

### Memory Management

```python
# ❌ ПЛОХО: утечка памяти при batch processing
for video in large_video_list:
    config = FIOConfig()  # Создавать каждый раз - OK
    features = extract_features(video, config=config)  # НО не очищать

# ✅ ХОРОШО: явная очистка
config = FIOConfig()
for video in large_video_list:
    features = extract_features(video, config=config)
    # ... обработать features ...
    del features  # Освободить память
    import gc; gc.collect()  # Явный GC при необходимости
```

### Validation Before Production

```python
# Всегда валидировать перед использованием
config = FIOConfig.from_json('user_config.json')

if not config.is_valid():
    errors = config.validate()
    for section, errs in errors.items():
        if errs:
            print(f"[{section}]")
            for err in errs:
                print(f"  ❌ {err}")
    raise ValueError("Invalid configuration")

# Безопасно использовать
features = extract_features('video.mp4', config=config)
```

### Reproducibility

```python
# Фиксировать random seed для reproducibility
config = FIOConfig()
config.statistics.random_seed = 42  # Или любое число

# Сохранить config вместе с результатами
results = {
    'features': extract_features('video.mp4', config=config),
    'config': config.to_dict(),
    'timestamp': time.time(),
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## 📊 PERFORMANCE TUNING GUIDE

### Скорость vs Качество

| Параметр | Fast ⚡ | Balanced ⚖️ | High Quality 🎯 |
|----------|---------|-------------|-----------------|
| `fps_target` | 6 | 12 | 20 |
| `max_frames` | 300 | 900 | 1500 |
| `std_roi_side` | 128 | 256 | 512 |
| `sample_rate_frames` | 15 | 6 | 3 |
| `bootstrap_n_samples` | 100 | 250 | 1000 |
| `enable_surrogate_test` | False | True | True |

### Memory Footprint

| Параметр | Impact |
|----------|--------|
| `std_roi_side` ↑ | Quadratic memory ↑ (256→512 = 4x RAM) |
| `max_frames` ↑ | Linear memory ↑ |
| `bootstrap_n_samples` ↑ | Linear temp memory ↑ |
| `use_mediapipe = True` | +100MB constant |

### Processing Time Estimates

Для типичного 1080p видео на Intel i7:

```
High Quality:    120-180 sec per minute of video
Balanced:        40-60 sec per minute of video
Fast:            15-25 sec per minute of video
Mobile:          8-12 sec per minute of video
```

---

## 🐛 TROUBLESHOOTING

### Проблема: OutOfMemoryError

```python
# Решение 1: Уменьшить ROI size
config.roi.std_roi_side = 128  # Вместо 512

# Решение 2: Сэмплировать реже
config.frequency.sample_rate_frames = 20  # Каждый 20-й кадр

# Решение 3: Меньше кадров
config.video.max_frames = 300
```

### Проблема: Слишком медленно

```python
# Используйте fast preset
config = ConfigPresets.production_fast()

# Или отключите тяжелые компоненты
config.statistics.enable_bootstrap_ci = False
config.statistics.enable_surrogate_test = False
```

### Проблема: Низкое качество детекции

```python
# Увеличить качество DFA/boxcount
config.fractal.dfa_min_rsquared = 0.95
config.fractal.boxcount_min_rsquared = 0.90

# Больше данных
config.video.fps_target = 15
config.frequency.sample_rate_frames = 3
```

---

## 📖 API REFERENCE

Полная документация классов конфигурации:

- `FIOConfig`: Мастер-контейнер всех параметров
- `VideoConfig`: Параметры обработки видео
- `ROIConfig`: Параметры извлечения ROI
- `FractalConfig`: Параметры фрактальных признаков (DFA, boxcount)
- `FrequencyConfig`: Параметры частотных артефактов
- `StatisticsConfig`: Параметры статистического анализа
- `TrainingConfig`: Параметры обучения моделей
- `ConfigPresets`: Готовые наборы параметров

---

**Автор:** Игорь (ORCID: 0009-0007-4607-1946)  
**Лицензия:** MIT  
**Версия:** 0.5.2  
**Дата:** 2026-01-18

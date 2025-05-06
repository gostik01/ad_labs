import dash
import numpy as np
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, ctx
from scipy.signal import butter, filtfilt

# Посилання на додаткові стилі для узгодження вигляду компонентів Dash та Bootstrap
DBC_CSS = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"

# Ініціалізація Dash-додатку з темою Bootstrap та додатковими стилями
# Завдання 1.1: Створення програми, яка використовує бібліотеки Matplotlib (тут Dash/Plotly як сучасна альтернатива) для створення графічного інтерфейсу.
# Завдання 3.1: Реалізація завдання 1 за допомогою сучасних графічних бібліотек (Plotly через Dash).
app = dash.Dash(
    __name__,
    title="Harmonic Visualisation (Dash)",
    external_stylesheets=[dbc.themes.BOOTSTRAP, DBC_CSS]
)

# Глобальні константи для генерації сигналів
T_MAX = 5.0          # Максимальний час (секунди) для генерації сигналу
N_POINTS = 1000      # Кількість точок для генерації даних сигналу
SAMPLING_RATE = N_POINTS / T_MAX # Частота дискретизації (точок/сек), важлива для фільтрації
NYQUIST_FREQ = 0.5 * SAMPLING_RATE # Частота Найквіста (макс. частота для коректної фільтрації без аліасингу)

# Словник з початковими значеннями та конфігурацією параметрів для UI
# Завдання 1.4: Програма повинна мати початкові значення кожного параметру.
DEFAULTS = {
    # Параметри чистої гармоніки (синусоїди) - Завдання 1.2 (параметри amplitude, frequency, phase)
    'amplitude': {'value': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1, 'category': 'function', 'label': 'Амплітуда'},
    'frequency': {'value': 2.0, 'min': 0.1, 'max': 20.0, 'step': 0.1, 'category': 'function', 'label': 'Частота (Hz)'},
    'phase': {'value': 0.0, 'min': -np.pi, 'max': np.pi, 'step': np.pi/16, 'category': 'function', 'label': 'Фаза (rad)'},
    # Параметри Гаусівського шуму - Завдання 1.2 (параметри noise_mean, noise_covariance (тут variance), show_noise)
    'noise_mean': {'value': 0.0, 'min': -0.5, 'max': 0.5, 'step': 0.05, 'category': 'noise', 'label': 'Шум: Середнє'},
    'noise_variance': {'value': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'category': 'noise', 'label': 'Шум: Дисперсія'},
    # Параметри фільтра Баттерворта (Butterworth) - Завдання 2.3 (параметри фільтру)
    'filter_order': {'value': 4, 'min': 1, 'max': 10, 'step': 1, 'category': 'filter', 'label': 'Фільтр (Butter): Порядок'},
    'filter_cutoff': {'value': 5.0, 'min': 0.1, 'max': NYQUIST_FREQ - 0.1, 'step': 0.1, 'category': 'filter', 'label': 'Фільтр (Butter): Зріз (Hz)'},
    # Параметри власного фільтра (ковзне середнє) - Завдання 3.2 (параметри фільтру)
    'custom_filter_window': {'value': 11, 'min': 3, 'max': 51, 'step': 2, 'category': 'custom_filter', 'label': 'Фільтр (Custom): Вікно'},
}

# Корекція початкового значення частоти зрізу, щоб вона була в допустимих межах (менше частоти Найквіста)
if DEFAULTS['filter_cutoff']['value'] >= NYQUIST_FREQ:
    DEFAULTS['filter_cutoff']['value'] = NYQUIST_FREQ * 0.9 # Встановлюємо на 90% від частоти Найквіста, якщо перевищено
elif DEFAULTS['filter_cutoff']['value'] <= 0:
     DEFAULTS['filter_cutoff']['value'] = 0.1 # Мінімальне значення, якщо встановлено 0 або менше

# Створення масиву часових позначок для сигналу
t = np.linspace(0, T_MAX, N_POINTS, endpoint=False)

# Простий кеш для зберігання згенерованого шуму
# Завдання 1.6: "...якщо ви змінили параметри гармоніки, але не змінювали параметри шуму, то шум має залишитись таким як і був, а не генеруватись наново."
noise_cache = {} # Формат кешу: { (mean, std_dev, size): noise_array }

# --- Функції Генерації та Обробки Сигналів ---

# Завдання 1.2: Реалізуйте функцію harmonic_with_noise... (ця функціональність розподілена)
# Ця функція відповідає за генерацію чистої гармоніки
def harmonic(time, amplitude, frequency, phase):
    """Генерує чисту гармоніку (синусоїду) за заданими параметрами."""
    # amplitude - амплітуда гармоніки.
    # frequency - частота гармоніки.
    # phase – фазовий зсув гармоніки
    return amplitude * np.sin(2 * np.pi * frequency * time + phase)

# Ця функція відповідає за генерацію шуму
def generate_noise(size, mean, variance):
    """Генерує Гаусівський шум з кешуванням для уникнення повторної генерації."""
    # mean - середнє шуму (noise_mean).
    # variance – дисперсія шуму (noise_covariance).
    std_dev = np.sqrt(max(0, variance)) # Стандартне відхилення є коренем з дисперсії; max(0, variance) для уникнення помилок з від'ємною дисперсією.
    key = (round(mean, 5), round(std_dev, 5), size) # Створюємо ключ для кешу на основі параметрів шуму.
    if key not in noise_cache:
        # Якщо шум з такими параметрами ще не генерувався, створюємо його і зберігаємо в кеш.
        noise_cache[key] = np.random.normal(mean, std_dev, size)
    return noise_cache[key]

# Розраховує коефіцієнти (b, a) низькочастотного фільтра Баттерворта
# Завдання 2.1: Отриману гармоніку з накладеним на неї шумом відфільтруйте за допомогою фільтру на ваш вибір (наприклад scipy.signal.iirfilter...).
def design_butter_lowpass_filter_ba(cutoff, fs, order=5):
    """Розраховує коефіцієнти (b, a) для низькочастотного фільтра Баттерворта."""
    nyq = 0.5 * fs # Частота Найквіста
    safe_cutoff = np.clip(cutoff, 0.01, nyq * 0.99) # Обмежуємо частоту зрізу безпечними межами.
    if not np.isclose(safe_cutoff, cutoff):
        print(f"Warning: Cutoff frequency {cutoff:.2f} Hz adjusted to {safe_cutoff:.2f} Hz to be within (0, Nyquist frequency).")
    normal_cutoff = safe_cutoff / nyq # Нормалізована частота зрізу (від 0 до 1).
    try:
        # Розрахунок коефіцієнтів фільтра 'b' (числильник) та 'a' (знаменник).
        b, a = butter(order, normal_cutoff, btype='low', analog=False, output='ba')
        return b, a
    except ValueError as e:
        print(f"Error designing Butterworth filter: {e}. Check filter order and cutoff frequency.")
        return None, None

# Застосовує цифровий фільтр (за коефіцієнтами b, a) до даних
def apply_filter_ba(data, b, a):
    """Застосовує цифровий фільтр (за коефіцієнтами b, a) до вхідних даних."""
    # Перевірка, чи можна застосувати фільтр (коефіцієнти розраховані, достатня довжина даних).
    # filtfilt вимагає, щоб len(data) > 3 * max(len(b), len(a)-1)
    if b is None or a is None or len(data) <= 3 * (max(len(b), len(a)) -1) :
        return np.copy(data) # Повертаємо копію вихідних даних, якщо фільтрація неможлива.
    try:
        # Застосування фільтра filtfilt (фільтрує вперед і назад для нульового фазового зсуву).
        return filtfilt(b, a, data)
    except Exception as e:
        # Обробка можливих помилок під час фільтрації.
        print(f"Error applying Butterworth filter: {e}")
        return np.copy(data)

# Завдання 3.2: Реалізуйте ваш власний фільтр, використовуючи виключно Python (а також numpy...).
def custom_moving_average_filter(signal, window_size):
    """Реалізує простий фільтр ковзного середнього за допомогою numpy.convolve."""
    # Перевірка мінімального розміру вікна.
    if window_size < 3:
        print("Warning: Custom filter window size < 3. Returning original signal.")
        return np.copy(signal)
    # Переконуємося, що розмір вікна непарний для симетричного усереднення.
    if window_size % 2 == 0:
        window_size += 1
        print(f"Adjusted custom filter window size to odd value: {window_size}")

    # Створюємо вікно для згортки (однакові ваги для усереднення).
    window = np.ones(int(window_size)) / int(window_size) # Забезпечуємо, що window_size є цілим.
    try:
        # Використовуємо згортку (convolve) для ефективного обчислення ковзного середнього.
        # mode='same' гарантує, що вихідний масив має той самий розмір, що й вхідний,
        # обробляючи граничні ефекти за допомогою доповнення нулями (за замовчуванням).
        filtered_signal = np.convolve(signal, window, mode='same')
        return filtered_signal
    except Exception as e:
        # Обробка можливих помилок під час згортки.
        print(f"Error applying custom filter: {e}")
        return np.copy(signal)

# --- Функції Побудови Компонентів Інтерфейсу ---

# Завдання 1.3: Слайдери (sliders), які відповідають за амплітуду, частоту гармоніки, а також слайдери для параметрів шуму.
def build_slider(param_id: str, config: dict):
    """Створює компонент Dash Slider з підписом та налаштуваннями."""
    return html.Div([
        dbc.Label(config['label'], html_for=param_id, className="fw-bold"),
        dcc.Slider(
            id=param_id,                  # Унікальний ID для колбеків.
            min=config['min'],            # Мінімальне значення.
            max=config['max'],            # Максимальне значення.
            step=config['step'],          # Крок зміни.
            value=config['value'],        # Початкове значення (Завдання 1.4).
            marks=None,                   # Не показувати мітки на шкалі для чистоти інтерфейсу.
            tooltip={"placement": "bottom", "always_visible": True}, # Показувати поточне значення під повзунком.
            className="dbc p-1",          # Клас для стилізації Bootstrap.
            updatemode='drag'             # Оновлення значення під час перетягування повзунка, а не лише після відпускання.
        )
    ], id=f"{param_id}-div")

# Створює картку Bootstrap з набором слайдерів для певної категорії параметрів.
# Це допоміжна функція для організації UI.
def build_settings_card(title: str, category: str):
    """Створює картку Bootstrap з групою слайдерів для заданої категорії."""
    # Генеруємо список слайдерів для цієї категорії на основі DEFAULTS.
    sliders = [build_slider(name, config) for name, config in DEFAULTS.items() if config['category'] == category]
    # Створюємо картку Bootstrap.
    return dbc.Card([
        dbc.CardHeader(title), # Заголовок картки.
        dbc.CardBody(dbc.Form(sliders, className='d-flex flex-column gap-3')) # Тіло картки зі слайдерами, розташованими вертикально.
    ])

# Макет Додатку

# Завдання 1.3: У програмі має бути створено головне вікно з такими елементами інтерфейсу:
# (Поле для графіку функції (plot), Слайдери, Чекбокс, Кнопка «Reset»)
app.layout = dbc.Container([
    # Заголовок сторінки
    dbc.Row(dbc.Col(html.H3("Інтерактивна візуалізація гармоніки з фільтрацією (Dash)"), width=12), className="mb-4 mt-3 text-center"),

    # Завдання 1.8: Залиште коментарі та інструкції для користувача, які пояснюють, як користуватися програмою.
    dbc.Row(dbc.Col(dbc.Alert([
        html.H5("Інструкція з використання:", className="alert-heading"),
        html.P("1. Використовуйте слайдери для налаштування параметрів чистої гармоніки (амплітуда, частота, фаза)."),
        html.P("2. Налаштуйте параметри шуму (середнє, дисперсія) та ввімкніть його відображення за допомогою перемикача 'Показати шум'."),
        html.P("3. Активуйте та налаштуйте фільтри (Butterworth, Власний) за допомогою відповідних перемикачів та слайдерів."),
        html.P("4. Графіки оновлюються автоматично під час перетягування будь-якого слайдера."),
        html.P("5. Кнопка 'Reset' скидає всі параметри до початкових значень."),
        html.P("6. Кнопка 'Randomise' встановлює випадкові значення для всіх параметрів (в межах їх діапазонів)."),
        html.P("7. Виберіть режим відображення графіків ('Комбінований' або 'Окремі') у спадному меню 'Режим перегляду'.")
    ], color="info"), width=12), className="mb-4"),

    # Ряд з картками налаштувань (слайдери)
    dbc.Row([
        # Картка для параметрів гармоніки (Завдання 1.3 - слайдери)
        dbc.Col(build_settings_card("Параметри гармоніки", "function"), md=6, lg=3),
        # Картка для параметрів шуму (Завдання 1.3 - слайдери)
        dbc.Col(build_settings_card("Параметри шуму", "noise"), md=6, lg=3),
        # Картка для фільтра Butterworth (Завдання 2.3 - інтерактивні елементи для фільтра)
        dbc.Col(build_settings_card("Фільтр Butterworth", "filter"), md=6, lg=3),
        # Картка для власного фільтра (Завдання 3.2 - інтерактивні елементи для фільтра)
        dbc.Col(build_settings_card("Власний фільтр", "custom_filter"), md=6, lg=3),
    ], className="mb-4"), # mb-4: margin bottom 4 для відступу

    # Картка з загальними налаштуваннями та кнопками керування
    dbc.Card([
        dbc.CardHeader("Загальні налаштування та Керування"),
        dbc.CardBody([
            dbc.Row([
                # Колонка з перемикачами (чекбокси)
                dbc.Col(dbc.Checklist(
                    id='switches-input',
                    options=[ # Опції для перемикачів
                        # Завдання 1.3, 1.5: Чекбокс для перемикання відображення шуму.
                        {"label": "Показати шум", "value": 'show_noise'},
                        # Завдання 2.3: Чекбокс показу для фільтра.
                        {"label": "Фільтр (Butter)", "value": 'show_filter_butter'},
                        # Завдання 3.2: Чекбокс показу для власного фільтра.
                        {"label": "Фільтр (Custom)", "value": 'show_filter_custom'},
                    ],
                    value=['show_noise', 'show_filter_butter'], # Початково ввімкнені шум та фільтр Butterworth.
                    inline=True, # Розміщувати перемикачі в один ряд.
                    switch=True, # Використовувати стиль "перемикачів" замість стандартних чекбоксів.
                ), md=6),
                # Колонка зі спадним меню та кнопками
                dbc.Col([
                    # Завдання 3.1: Спадне меню (drop-down menu).
                    html.Div([
                        dbc.Label("Режим перегляду:", html_for='view-mode-dropdown', className="me-2"),
                        dcc.Dropdown(
                            id='view-mode-dropdown',
                            options=[
                                {'label': 'Комбінований графік', 'value': 'combined'},
                                # Завдання 3.1: Додайте декілька вікон для візуалізації замість одного (реалізовано через опцію 'Окремі графіки').
                                {'label': 'Окремі графіки', 'value': 'individual'},
                            ],
                            value='combined', # Початкове значення - комбінований графік.
                            clearable=False, # Не можна очистити вибір (завжди щось вибрано).
                            style={'minWidth': '200px'} # Мінімальна ширина для кращого вигляду.
                        )], className="d-flex align-items-center mb-2"), # d-flex для горизонтального розміщення, mb-2 для відступу.
                    # Кнопки керування
                    html.Div([
                        # Завдання 3.1: Інші інтерактивні елементи на власний розсуд (кнопка Randomise).
                        dbc.Button("Randomise", id='random-button', color="primary", n_clicks=0, className="me-2"), # Кнопка випадкових значень.
                        # Завдання 1.3, 1.7: Кнопка «Reset», яка відновлює початкові параметри.
                        dbc.Button("Reset", id='reset-button', color="danger", n_clicks=0),
                     ], className="d-flex") # d-flex для горизонтального розміщення кнопок.
                ], md=6)
            ])
        ])
    ], className="mb-4"),

    # Завдання 1.3: Поле для графіку функції (plot).
    dbc.Row(id='graph-area')

], fluid=True, className="dbc") # fluid=True робить контейнер на всю ширину сторінки.


# Створює об'єкт фігури Plotly для відображення одного або кількох графіків.
def create_plotly_figure(title, x_data, y_data_dict):
    """Створює фігуру Plotly з заданими даними та налаштуваннями для візуалізації."""
    fig = go.Figure() # Створюємо порожню фігуру Plotly.
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] # Стандартні кольори Plotly для різних ліній.
    color_idx = 0
    # Додаємо кожен графік (лінію) до фігури.
    for name, y_data in y_data_dict.items():
        fig.add_trace(go.Scatter(
            x=x_data,         # Дані для осі X (час).
            y=y_data,         # Дані для осі Y (амплітуда).
            mode='lines',     # Малювати лініями.
            name=name,        # Назва лінії (для легенди).
            line=dict(color=colors[color_idx % len(colors)]) # Встановлюємо колір лінії, циклічно вибираючи з палітри.
        ))
        color_idx += 1

    # Налаштування вигляду фігури (заголовок, осі, легенда, відступи).
    fig.update_layout(
        title=title, # Заголовок графіка.
        xaxis_title='Час (t)', # Підпис осі X.
        yaxis_title='Амплітуда (y)', # Підпис осі Y.
        margin=dict(l=40, r=20, t=40, b=30), # Зменшені відступи для економії місця.
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01), # Положення легенди (зверху зліва).
        uirevision='constant' # Зберігає масштаб та положення графіка при оновленні даних (важливо для плавних оновлень під час 'drag').
    )
    return fig # Повертаємо сконфігуровану фігуру.

# Колбек для керування станом (ввімкнено/вимкнено) та стилем (прозорістю) слайдерів.
@app.callback(
    # Output: Оновлюємо стиль (прозорість) та стан disabled для КОЖНОГО div-контейнера слайдера та самого слайдера.
    [Output(f"{param_id}-div", "style") for param_id in DEFAULTS] +
    [Output(f"{param_id}", "disabled") for param_id in DEFAULTS],
    # Input: Залежить від стану перемикачів (чекбоксів).
    Input('switches-input', 'value')
)
def toggle_sliders_visibility(switches):
    """Керує видимістю (через прозорість) та активністю слайдерів залежно від стану відповідних перемикачів."""
    # Перевіряємо, які перемикачі зараз увімкнені.
    show_noise = 'show_noise' in switches
    show_filter_butter = 'show_filter_butter' in switches
    show_filter_custom = 'show_filter_custom' in switches

    disabled_states = {} # Словник для зберігання стану 'disabled' кожного слайдера.
    styles = {}          # Словник для зберігання стилю кожного div-контейнера слайдера.
    opacity_off = {'opacity': '0.4'} # Стиль для "вимкненого" стану (напівпрозорий).
    opacity_on = {'opacity': '1'}    # Стиль для "ввімкненого" стану (непрозорий).

    # Визначаємо стиль та стан 'disabled' для кожного слайдера залежно від активних перемикачів.
    for param_id, config in DEFAULTS.items():
        category = config['category'] # Категорія параметра (function, noise, filter, custom_filter).
        is_disabled = False # За замовчуванням слайдер активний.
        style = opacity_on  # За замовчуванням слайдер непрозорий.

        # Вимикаємо слайдери шуму, якщо перемикач "Показати шум" вимкнено.
        if category == 'noise' and not show_noise:
            is_disabled = True
            style = opacity_off
        # Вимикаємо слайдери фільтра Butterworth, якщо його перемикач вимкнений.
        elif category == 'filter' and not show_filter_butter:
            is_disabled = True
            style = opacity_off
        # Вимикаємо слайдери власного фільтра, якщо його перемикач вимкнений.
        elif category == 'custom_filter' and not show_filter_custom:
            is_disabled = True
            style = opacity_off

        disabled_states[param_id] = is_disabled
        styles[f"{param_id}-div"] = style

    # Повертаємо списки стилів та станів 'disabled' у правильному порядку, як визначено в Output.
    style_list = [styles[f"{param_id}-div"] for param_id in DEFAULTS]
    disabled_list = [disabled_states[param_id] for param_id in DEFAULTS]

    return style_list + disabled_list # Повертаємо об'єднаний список для всіх Output.

# Колбек для обробки натискань на кнопки "Reset" та "Randomise".
@app.callback(
    # Output: Оновлюємо значення КОЖНОГО слайдера.
    [Output(param_id, 'value') for param_id in DEFAULTS],
    # Input: Залежить від кількості натискань на кнопки.
    [Input('reset-button', 'n_clicks'),
     Input('random-button', 'n_clicks')]
)
def handle_buttons(reset_clicks, random_clicks):
    """Обробляє натискання кнопок 'Reset' та 'Randomise', оновлюючи значення слайдерів."""
    triggered_id = ctx.triggered_id # Визначаємо, яка саме кнопка (Input) спричинила виклик колбеку.

    if triggered_id == 'reset-button':
        # Завдання 1.7: Після натискання кнопки «Reset», мають відновитись початкові параметри.
        # Повертаємо список початкових значень з DEFAULTS для кожного слайдера.
        return [config['value'] for config in DEFAULTS.values()]
    elif triggered_id == 'random-button':
        # Завдання 3.1 (додатковий інтерактивний елемент): кнопка Randomise.
        # Генеруємо випадкові значення для кожного параметра в межах його min/max.
        new_values = []
        for config in DEFAULTS.values():
            if config['step'] == 1: # Для цілочисельних параметрів (наприклад, порядок фільтра).
                 val = np.random.randint(config['min'], config['max'] + 1)
            elif config['step'] == 2: # Для кроку 2 (наприклад, розмір вікна, гарантуємо непарність).
                 # Генеруємо випадкове непарне число в діапазоні.
                 val = np.random.randint(config['min'] // 2, (config['max'] + 1) // 2) * 2 + config['min'] % 2
                 val = np.clip(val, config['min'], config['max']) # Переконуємось, що не виходимо за межі.
            else: # Для дійсних параметрів.
                 val = np.random.uniform(config['min'], config['max'])
            new_values.append(val)
        return new_values # Повертаємо список нових випадкових значень.
    else:
        # Якщо колбек спрацював не через кнопку (наприклад, при першому завантаженні сторінки),
        # повертаємо початкові значення (Завдання 1.4).
        return [config['value'] for config in DEFAULTS.values()]

# Головний колбек: оновлює графіки при зміні будь-якого параметра або налаштування.
# Завдання 1.4: ...а також передавати параметри для відображення оновленого графіку.
# Завдання 1.6: Після оновлення параметрів програма повинна одразу оновлювати графік функції...
# Завдання 2.3: ...відфільтрована гармоніка має оновлюватись разом з початковою.
@app.callback(
    # Output: Оновлює вміст (дочірні елементи) області для графіків (dbc.Row(id='graph-area')).
    Output('graph-area', 'children'),
    # Input: Залежить від значень всіх слайдерів, стану перемикачів та режиму перегляду зі спадного меню.
    [Input(param_id, 'value') for param_id in DEFAULTS] + # Значення всіх слайдерів.
    [Input('switches-input', 'value'),                   # Активні перемикачі.
     Input('view-mode-dropdown', 'value')]                # Вибраний режим перегляду.
)
def update_graphs(*args):
    """
    Оновлює графіки на основі поточних значень слайдерів, перемикачів та режиму перегляду.
    Цей колбек викликається щоразу, коли змінюється будь-який з його Input.
    """
    # Розпаковуємо всі вхідні аргументи.
    num_params = len(DEFAULTS)
    param_values = args[:num_params] # Значення слайдерів.
    switches = args[num_params]     # Список активних перемикачів (значень value з Checklist).
    view_mode = args[num_params + 1] # Вибраний режим ('combined' або 'individual').

    # Створюємо словник поточних значень параметрів для зручності доступу за назвою.
    current_params = {name: val for name, val in zip(DEFAULTS.keys(), param_values)}

    # Перевіряємо стан перемикачів.
    # Завдання 1.2 (параметр show_noise), Завдання 1.5: Через чекбокс користувач може вмикати або вимикати відображення шуму...
    show_noise = 'show_noise' in switches
    # Завдання 2.3: Додайте відповідні інтерактивні елементи (чекбокс показу...).
    show_filter_butter = 'show_filter_butter' in switches
    # Завдання 3.2: (чекбокс для власного фільтра).
    show_filter_custom = 'show_filter_custom' in switches

    # --- Генерація сигналів ---
    # 1. Чиста гармоніка
    # Використовуємо розпакування словника для передачі параметрів категорії 'function' до функції harmonic.
    y_pure = harmonic(t, **{k: v for k, v in current_params.items() if DEFAULTS[k]['category'] == 'function'})

    # 2. Шум та зашумлена гармоніка
    # Завдання 1.6: "...якщо ви змінили параметри шуму, змінюватись має лише шум – параметри гармоніки мають залишатись незмінними."
    # Це досягається тим, що `generate_noise` викликається з поточними параметрами шуму. Якщо вони не змінилися, повернеться кешований шум.
    noise_params = {k: v for k, v in current_params.items() if DEFAULTS[k]['category'] == 'noise'}
    noise = generate_noise(N_POINTS, noise_params['noise_mean'], noise_params['noise_variance']) if show_noise else np.zeros_like(t)
    y_noisy = y_pure + noise # Зашумлений сигнал = чиста гармоніка + шум.

    # --- Фільтрація сигналів ---
    # 3. Фільтрація фільтром Баттерворта (якщо увімкнено)
    y_filtered_butter = np.copy(y_noisy) # За замовчуванням - зашумлений сигнал (якщо фільтр вимкнено).
    if show_filter_butter:
        filter_params_butter = {k: v for k, v in current_params.items() if DEFAULTS[k]['category'] == 'filter'}
        b, a = design_butter_lowpass_filter_ba(filter_params_butter['filter_cutoff'], SAMPLING_RATE, int(filter_params_butter['filter_order']))
        if b is not None and a is not None: # Якщо коефіцієнти успішно розраховані.
            y_filtered_butter = apply_filter_ba(y_noisy, b, a)
            # Завдання 2.1: Відфільтрована гармоніка має бути максимально близька до «чистої». (Залежить від налаштувань фільтра)

    # 4. Фільтрація власним фільтром (ковзне середнє) (якщо увімкнено)
    y_filtered_custom = np.copy(y_noisy) # За замовчуванням - зашумлений сигнал.
    if show_filter_custom:
        filter_params_custom = {k: v for k, v in current_params.items() if DEFAULTS[k]['category'] == 'custom_filter'}
        y_filtered_custom = custom_moving_average_filter(y_noisy, int(filter_params_custom['custom_filter_window']))

    # --- Формування графіків для відображення ---
    graphs_to_display = {} # Словник для зберігання фігур Plotly.
    if view_mode == 'combined':
        # Створюємо один комбінований графік.
        combined_data = {'Чиста гармоніка': y_pure} # Завжди показуємо чисту гармоніку.
        # Завдання 1.5: Якщо прапорець прибрано – відображати «чисту гармоніку», якщо ні – зашумлену.
        if show_noise:
            combined_data['Зашумлена гармоніка'] = y_noisy
        # Завдання 2.2: Відобразіть відфільтровану «чисту» гармоніку поряд з початковою.
        if show_filter_butter:
            combined_data['Фільтр Butterworth'] = y_filtered_butter
        # Завдання 3.2: Застосуйте фільтр (і відобразіть результат).
        if show_filter_custom:
            combined_data['Власний фільтр (ковз. середнє)'] = y_filtered_custom
        # Створюємо фігуру Plotly для комбінованого графіка.
        graphs_to_display['combined'] = create_plotly_figure("Комбінований графік сигналів", t, combined_data)

    elif view_mode == 'individual':
        # Завдання 3.1: Додайте декілька вікон для візуалізації замість одного.
        # Створюємо окремі графіки для кожного активного сигналу.
        graphs_to_display['pure'] = create_plotly_figure("Чиста гармоніка", t, {'Гармоніка': y_pure})
        if show_noise:
            graphs_to_display['noisy'] = create_plotly_figure("Зашумлена гармоніка", t, {'Зашумлений сигнал': y_noisy})
        if show_filter_butter:
            # Завдання 2.2: Відобразіть відфільтровану «чисту» гармоніку поряд з початковою (тут в окремому вікні).
            graphs_to_display['filtered_butter'] = create_plotly_figure("Результат фільтра Butterworth", t, {'Відфільтровано (Butterworth)': y_filtered_butter})
        if show_filter_custom:
            # Завдання 3.2: (відображення результату власного фільтра).
            graphs_to_display['filtered_custom'] = create_plotly_figure("Результат власного фільтра", t, {'Відфільтровано (Custom)': y_filtered_custom})

    # --- Формування Layout для області графіків (розміщення їх у колонки Bootstrap) ---
    graph_cols = [] # Список колонок Bootstrap, кожна з яких містить графік.
    num_graphs = len(graphs_to_display)
    # Розрахунок ширини колонок Bootstrap для адаптивного розміщення графіків (1, 2 або 3 графіки в ряд).
    col_width = 12 # За замовчуванням, якщо 1 графік (на всю ширину).
    if num_graphs == 2: col_width = 6 # Два графіки в ряд (кожен по 6 колонок з 12).
    if num_graphs == 3: col_width = 4 # Три графіки в ряд (кожен по 4 колонки).
    if num_graphs >= 4: col_width = 6 # Якщо 4 або більше, то по 2 графіки в ряд.

    # Створюємо колонку Bootstrap з компонентом dcc.Graph для кожної фігури.
    for i, (graph_id, fig) in enumerate(graphs_to_display.items()):
        graph_cols.append(
            dbc.Col( # Колонка Bootstrap.
                dcc.Graph( # Компонент Dash для відображення графіка Plotly.
                    id=f"graph-{graph_id}-{i}", # Унікальний ID для графіка.
                    figure=fig, # Фігура Plotly для відображення.
                    config={'staticPlot': False} # Дозволяємо інтерактивність (масштабування, панорамування тощо).
                ),
                width=col_width # Встановлюємо ширину колонки.
            )
        )

    return graph_cols

# Точка входу: запуск веб-сервера Dash.
if __name__ == "__main__":
    app.run(debug=True)
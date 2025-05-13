import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Глобальні константи та початкові значення ---
T_MAX = 5.0
N_POINTS = 1000
time_array = np.linspace(0, T_MAX, N_POINTS, endpoint=False)

# Завдання 1.4: Програма повинна мати початкові значення кожного параметру.
DEFAULTS = {
    'amplitude': {'value': 1.0, 'min': 0.1, 'max': 5.0, 'step': 0.1, 'label': 'Амплітуда:'},
    'frequency': {'value': 2.0, 'min': 0.1, 'max': 20.0, 'step': 0.1, 'label': 'Частота (Hz):'},
    'phase': {'value': 0.0, 'min': -np.pi, 'max': np.pi, 'step': np.pi/16, 'label': 'Фаза (rad):'},
    'noise_mean': {'value': 0.0, 'min': -0.5, 'max': 0.5, 'step': 0.05, 'label': 'Шум: Середнє:'},
    'noise_covariance': {'value': 0.1, 'min': 0.0, 'max': 1.0, 'step': 0.01, 'label': 'Шум: Дисперсія:'}, # 'noise_covariance' тут як дисперсія для 1D
}

# Простий кеш для зберігання згенерованого шуму
# Завдання 1.6: "...шум має залишитись таким як і був, а не генеруватись наново."
noise_cache = {}

# --- Функції Генерації Сигналів ---

def harmonic(t_arr, amplitude, frequency, phase):
    """Генерує чисту гармоніку."""
    return amplitude * np.sin(2 * np.pi * frequency * t_arr + phase)

def generate_noise(size, mean, variance):
    """Генерує Гаусівський шум з кешуванням."""
    std_dev = np.sqrt(max(0, variance))
    key = (round(mean, 5), round(std_dev, 5), size)
    if key not in noise_cache:
        noise_cache[key] = np.random.normal(mean, std_dev, size)
    return noise_cache[key]

# Завдання 1.2: Реалізуйте функцію harmonic_with_noise...
def harmonic_with_noise(t_arr, amplitude, frequency, phase, noise_mean, noise_covariance_param, show_noise_flag):
    """
    Генерує гармоніку, опціонально з накладеним Гаусівським шумом.
    Повертає сигнал та назву для легенди.
    """
    y_pure = harmonic(t_arr, amplitude, frequency, phase)
    if show_noise_flag:
        # 'noise_covariance_param' використовується як дисперсія для 1D шуму
        current_noise = generate_noise(len(t_arr), noise_mean, noise_covariance_param)
        return y_pure + current_noise, "Гармоніка з шумом"
    else:
        return y_pure, "Чиста гармоніка"

# --- Клас GUI Додатку ---
# Завдання 1.1: Створення програми, яка використовує бібліотеки Matplotlib для створення графічного інтерфейсу.
class HarmonicApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Harmonic Visualisation (Matplotlib + Tkinter)")

        # --- Створення елементів GUI ---
        # Завдання 1.3: У програмі має бути створено головне вікно з такими елементами інтерфейсу:
        
        # Рамка для слайдерів
        controls_frame = ttk.Frame(self.root, padding="10")
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Словник для зберігання Tkinter Variables
        self.tk_vars = {}

        # Завдання 1.3: ...Слайдери (sliders)...
        param_order = ['amplitude', 'frequency', 'phase', 'noise_mean', 'noise_covariance']
        for i, param_name in enumerate(param_order):
            config = DEFAULTS[param_name]
            ttk.Label(controls_frame, text=config['label']).grid(row=i*2, column=0, sticky=tk.W, pady=(5,0))
            
            var = tk.DoubleVar(value=config['value'])
            self.tk_vars[param_name] = var
            
            slider = ttk.Scale(
                controls_frame,
                from_=config['min'],
                to=config['max'],
                orient=tk.HORIZONTAL,
                variable=var,
                length=200,
                command=self.update_plot_on_event # Оновлення при перетягуванні
            )
            slider.grid(row=i*2+1, column=0, sticky=tk.EW, pady=(0,10))
            
            # Додаємо мітку для відображення поточного значення слайдера
            value_label = ttk.Label(controls_frame, text=f"{var.get():.2f}")
            value_label.grid(row=i*2+1, column=1, sticky=tk.W, padx=5)
            self.tk_vars[f"{param_name}_label"] = value_label # Зберігаємо мітку для оновлення

        # Завдання 1.3: ...Чекбокс для перемикання відображення шуму
        self.show_noise_var = tk.BooleanVar(value=True) # Параметр show_noise з Завдання 1.2
        noise_checkbox = ttk.Checkbutton(
            controls_frame,
            text="Показати шум",
            variable=self.show_noise_var,
            command=self.update_plot_on_event # Оновлення при зміні стану
        )
        noise_checkbox.grid(row=len(param_order)*2, column=0, columnspan=2, sticky=tk.W, pady=10)

        # Завдання 1.3: ...Кнопка «Reset»
        # Завдання 1.7: Після натискання кнопки «Reset», мають відновитись початкові параметри
        reset_button = ttk.Button(
            controls_frame,
            text="Reset",
            command=self.reset_parameters
        )
        reset_button.grid(row=len(param_order)*2 + 1, column=0, columnspan=2, pady=10)

        # Завдання 1.8: Коментарі та інструкції
        instructions_frame = ttk.Frame(controls_frame, padding="5")
        instructions_frame.grid(row=len(param_order)*2 + 2, column=0, columnspan=2, pady=10, sticky=tk.EW)
        ttk.Label(instructions_frame, text="Інструкції:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        instructions_text = (
            "1. Використовуйте слайдери для зміни параметрів.\n"
            "2. Графік оновлюється автоматично.\n"
            "3. 'Показати шум' вмикає/вимикає шум.\n"
            "4. 'Reset' скидає параметри до початкових."
        )
        ttk.Label(instructions_frame, text=instructions_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # --- Налаштування області для графіку Matplotlib ---
        # Завдання 1.3: Поле для графіку функції (plot)
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(7, 5)) # Зменшив розмір для кращого вміщення
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.fig.tight_layout(pad=2.0) # Додає трохи відступів

        # Початкове відображення графіку
        self.update_plot()

    def update_slider_value_labels(self):
        """Оновлює текстові мітки біля слайдерів."""
        for param_name in ['amplitude', 'frequency', 'phase', 'noise_mean', 'noise_covariance']:
            if f"{param_name}_label" in self.tk_vars and param_name in self.tk_vars:
                try:
                    value = self.tk_vars[param_name].get()
                    self.tk_vars[f"{param_name}_label"].config(text=f"{value:.2f}")
                except tk.TclError: # Може виникнути, якщо віджет ще не повністю створений
                    pass

    # Завдання 1.4: ...а також передавати параметри для відображення оновленого графіку.
    # Завдання 1.6: Після оновлення параметрів програма повинна одразу оновлювати графік...
    def update_plot(self):
        """Оновлює графік на основі поточних значень параметрів."""
        amp = self.tk_vars['amplitude'].get()
        freq = self.tk_vars['frequency'].get()
        phase_val = self.tk_vars['phase'].get()
        noise_m = self.tk_vars['noise_mean'].get()
        noise_cov = self.tk_vars['noise_covariance'].get()
        show_n_flag = self.show_noise_var.get() # Завдання 1.5: ...відображати «чисту гармоніку», якщо ні – зашумлену.

        # Завдання 1.6 (про кешування шуму) обробляється всередині generate_noise,
        # яка викликається з harmonic_with_noise
        y_signal, signal_label = harmonic_with_noise(
            time_array, amp, freq, phase_val, noise_m, noise_cov, show_n_flag
        )

        self.ax.clear()
        self.ax.plot(time_array, y_signal, label=signal_label)
        self.ax.set_title("Гармоніка")
        self.ax.set_xlabel("Час (t)")
        self.ax.set_ylabel("Амплітуда (y)")
        self.ax.legend(loc='upper right')
        self.ax.grid(True)
        self.canvas.draw_idle() # Використовуємо draw_idle для ефективного оновлення

        self.update_slider_value_labels() # Оновлюємо мітки значень слайдерів

    def update_plot_on_event(self, event=None): # event=None для сумісності з command
        """Обгортка для update_plot, викликається при зміні слайдера/чекбокса."""
        self.update_plot()

    def reset_parameters(self):
        """Скидає всі параметри до початкових значень."""
        # Завдання 1.7: Після натискання кнопки «Reset», мають відновитись початкові параметри
        for param_name, config in DEFAULTS.items():
            if param_name in self.tk_vars:
                self.tk_vars[param_name].set(config['value'])
        
        self.show_noise_var.set(True) # Повертаємо чекбокс шуму до початкового стану
        self.update_plot() # Оновлюємо графік після скидання

# --- Запуск додатку ---
if __name__ == "__main__":
    root = tk.Tk()
    app = HarmonicApp(root)
    root.mainloop()
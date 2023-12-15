import tkinter as tk
from tkinter import filedialog, messagebox, font
import subprocess
import threading
from PIL import Image, ImageTk


def hide_image():
    image_label.grid_forget()
    result_label.grid_forget()

def train_model():
    train_button.config(state=tk.DISABLED)  # Desactivar el botón
    classify_button.config(state=tk.DISABLED)  # Desactivar el botón

    try:
        # Validar y obtener el número de epochs
        epochs = num_epochs_entry.get()
        if not epochs.isdigit() or int(epochs) <= 0:
            messagebox.showerror("Error", "Por favor, ingresa un número válido de epochs (entero positivo)")
            return
        loading_label.grid(row=3, column=0, columnspan=2, pady=5)  # Mostrar el icono de carga
        threading.Thread(target=lambda: run_training(epochs), daemon=True).start()
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error inesperado: {e}")
    finally:
        train_button.config(state=tk.NORMAL)  # Reactivar el botón
        classify_button.config(state=tk.NORMAL)  # Reactivar el botón

def run_training(epochs):
    try:
        hide_image()
        subprocess.run(['python', 'train.py', '--epochs', epochs], check=True)
        messagebox.showinfo("Información", f"Modelo entrenado con éxito")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error en el entrenamiento del modelo: {e}\n{e.stderr}")
    finally:
        loading_label.grid_forget()  # Ocultar el icono de carga
        train_button.config(state=tk.NORMAL)  # Reactivar el botón
        classify_button.config(state=tk.NORMAL)  # Reactivar el botón

def run_classification(file_path):
    try:
        loading_label.grid(row=4, column=0, columnspan=2, pady=5)  # Mostrar el icono de carga
        result = subprocess.run(['python', 'display.py', file_path], check=True, text=True, capture_output=True, encoding='utf-8')
        output_lines = result.stdout.strip().split('\n')
        last_line = output_lines[-1]
        classification_result.set(last_line)

    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error en la clasificación: {e}")
    finally:
        loading_label.grid_forget()  
        train_button.config(state=tk.NORMAL)  # Reactivar el botón
        classify_button.config(state=tk.NORMAL)  # Reactivar el botón
        display_image(file_path)  # Muestra la imagen seleccionada

def display_image(path):
    img = Image.open(path)
    img.thumbnail((200, 200))  # Redimensiona la imagen
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img  # Guarda una referencia
    image_label.grid(row=4, column=0, columnspan=2)  

def classify_image():
    train_button.config(state=tk.DISABLED)  # Desactivar el botón
    classify_button.config(state=tk.DISABLED)  # Desactivar el botón
    file_path = filedialog.askopenfilename()
    if file_path:
        loading_label.grid(row=3, column=0, columnspan=2, pady=5) 
        threading.Thread(target=lambda: run_classification(file_path), daemon=True).start()

root = tk.Tk()
root.title("Clasificador de Insectos")
root.geometry("400x300")

title_font = font.Font(family="Helvetica", size=16, weight="bold")
title_label = tk.Label(root, text="Insect Recognition", font=title_font)
title_label.grid(row=0, column=0, pady=10, columnspan=2)

loading_image = tk.PhotoImage(file='loading.gif')
loading_label = tk.Label(root, image=loading_image)
image_label = tk.Label(root)
image_label.grid(row=4, column=0, columnspan=2)

# Configurar los botones en la parte superior y centrados
button_width = 20
button_height = 2  

train_button = tk.Button(root, text="Entrenar Modelo", command=train_model, width=button_width, height=button_height)
train_button.grid(row=1, column=0, pady=5, padx=5, sticky="e")  # Alineación a la derecha

classify_button = tk.Button(root, text="Cargar y Clasificar Imagen", command=classify_image, width=button_width, height=button_height)
classify_button.grid(row=1, column=1, pady=5, padx=5, sticky="w")  # Alineación a la izquierda

classification_result = tk.StringVar()

epochs_label = tk.Label(root, text="Número de Epochs:")
epochs_label.grid(row=2, column=0, pady=5, sticky="e")  # Alineación a la derecha

num_epochs_entry = tk.Entry(root)
num_epochs_entry.grid(row=2, column=1, pady=5, padx=5, sticky="w")  # Alineación a la izquierda

result_label = tk.Label(root, textvariable=classification_result)
result_label.grid(row=3, column=0, columnspan=2, sticky="nsew")  

# Configurar el peso de las filas y columnas para que se expandan y centren los elementos
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=1)
root.grid_rowconfigure(2, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()

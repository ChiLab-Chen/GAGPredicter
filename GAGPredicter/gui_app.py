import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import sys
import pandas as pd
import math
import threading


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
except NameError:
    project_root = os.path.abspath('.')
    if project_root not in sys.path:
        sys.path.append(project_root)


try:
    from predictor.predictor import EnsemblePredictor
except ImportError as e:
    messagebox.showerror("Import Error", f"Could not import the predictor module. Please ensure the project structure is correct.\nError: {e}")
    sys.exit(1)


class CircularProgressbar(ctk.CTkFrame):
    def __init__(self, *args,
                 width=200,
                 height=200,
                 start_angle=90,
                 progress_width=15,
                 progress_color="#2D3436",
                 bg_color="#E0E0E0",
                 text_color="#000000",
                 font=("Arial", 20, "bold"),
                 **kwargs):
        super().__init__(*args, width=width, height=height, **kwargs)

        self.width = width
        self.height = height
        self.start_angle = start_angle
        self.progress_width = progress_width
        self.progress_color = progress_color
        self.bg_color = bg_color
        
        self.grid_propagate(False)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.canvas = ctk.CTkCanvas(self, background=self.cget("fg_color"), width=width, height=height, highlightthickness=0)
        self.canvas.grid(row=0, column=0)
        
        self.percentage_label = ctk.CTkLabel(self, text="0%", font=font, text_color=text_color)
        self.percentage_label.grid(row=0, column=0)
        
        self._animation_id = None
        self._current_angle = 0
        self.draw_bg()

    def draw_bg(self):
        self.canvas.create_arc(
            self.progress_width/2, self.progress_width/2,
            self.width - self.progress_width/2, self.height - self.progress_width/2,
            style="arc",
            width=self.progress_width,
            start=0,
            extent=360,
            outline=self.bg_color
        )

    def set_progress(self, value):
        self.canvas.delete("progress")
        self.percentage_label.configure(text=f"{int(value)}%")
        
        angle = (value / 100) * 359.99  # Use 359.99 to avoid Tkinter bug with 360 degrees
        
        self.canvas.create_arc(
            self.progress_width/2, self.progress_width/2,
            self.width - self.progress_width/2, self.height - self.progress_width/2,
            style="arc",
            width=self.progress_width,
            start=self.start_angle,
            extent=angle,
            outline=self.progress_color,
            tags="progress"
        )
        if value >= 100:
             # Draw a small circle at the end to make it look complete
            x = (self.width/2) + ((self.width/2 - self.progress_width/2) * math.cos(math.radians(self.start_angle)))
            y = (self.height/2) - ((self.height/2 - self.progress_width/2) * math.sin(math.radians(self.start_angle)))
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=self.progress_color, outline=self.progress_color, tags="progress")

    def start_animation(self):
        self.percentage_label.configure(text="...")
        self.canvas.delete("progress")
        self._update_animation()

    def _update_animation(self):
        self.canvas.delete("progress")
        self._current_angle += 10
        self.canvas.create_arc(
            self.progress_width/2, self.progress_width/2,
            self.width - self.progress_width/2, self.height - self.progress_width/2,
            style="arc",
            width=self.progress_width,
            start=self._current_angle,
            extent=120, # Arc length
            outline=self.progress_color,
            tags="progress"
        )
        self._animation_id = self.after(25, self._update_animation)

    def stop_animation(self):
        if self._animation_id:
            self.after_cancel(self._animation_id)
            self._animation_id = None
        self.set_progress(0)



ctk.set_appearance_mode("light")

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("GAGPredict Pro")
        self.geometry("1400x850")
        self.configure(fg_color="#F0F2F5")
        
        self.colors = {
            "primary": "#000000",
            "secondary": "#6A737D",
            "bg": "#F0F2F5",
            "card": "#FFFFFF",
            "nav_hover": "#E8E8E8",
            "nav_selected_bg": "#000000",
            "nav_selected_fg": "#FFFFFF",
            "button": "#2D3436",
            "button_hover": "#636E72",
        }
        self.fonts = {
            "title": ("Segoe UI", 22, "bold"),
            "card_title": ("Segoe UI", 16, "bold"),
            "body": ("Segoe UI", 14),
            "small": ("Segoe UI", 12),
            "icon": ("Arial", 18),
        }
        
        self.threshold = 0.5
        self.current_view = None
        self.nav_buttons = {}
        self.prediction_history = []
        self.batch_thread = None
        self.batch_status = {"done": False, "error": None}
        self.batch_input_path = None
        
        try:
            print("Loading model...")
            self.predictor = EnsemblePredictor(verbose=False)
            print("✅ Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Load Failed", f"Could not load the ensemble model. Please check model files.\nError: {e}")
            self.destroy()
            sys.exit(1)

        self.create_widgets()

    def create_widgets(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.create_nav_bar()
        
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(20, 10), pady=20)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.create_topbar()
        self.create_content_views()

        self.create_right_info_panel()

        self.select_view("single")


    def create_nav_bar(self):
        nav_frame = ctk.CTkFrame(self, width=80, fg_color=self.colors["card"], corner_radius=0)
        nav_frame.grid(row=0, column=0, sticky="nsw")
        nav_frame.pack_propagate(False)

        nav_items = {"single": "📝", "batch": "📂", "settings": "⚙️"}

        for i, (key, icon) in enumerate(nav_items.items()):
            button = ctk.CTkButton(
                nav_frame, text=icon, width=50, height=50, corner_radius=10,
                font=self.fonts["icon"], fg_color="transparent", text_color=self.colors["primary"],
                hover_color=self.colors["nav_hover"], command=lambda k=key: self.select_view(k)
            )
            button.pack(pady=15, padx=15)
            self.nav_buttons[key] = button


    def create_topbar(self):
        topbar = ctk.CTkFrame(self.main_frame, height=60, fg_color="transparent")
        topbar.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        self.view_title = ctk.CTkLabel(topbar, text="Dashboard", font=self.fonts["title"])
        self.view_title.pack(expand=True)
    
    def create_content_views(self):
        # --- Single Sequence View ---
        self.single_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.single_view.grid_columnconfigure(0, weight=1)
        
        input_card = ctk.CTkFrame(self.single_view, fg_color=self.colors["card"], corner_radius=12)
        input_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        input_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(input_card, text="Enter a single glycan sequence for real-time prediction.", font=self.fonts["body"], text_color=self.colors["secondary"]).grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))
        self.seq_entry = ctk.CTkEntry(input_card, placeholder_text="Enter sequence here...", height=40, font=self.fonts["body"])
        self.seq_entry.grid(row=1, column=0, sticky="ew", padx=20)
        predict_btn = ctk.CTkButton(input_card, text="Predict", height=40, command=self.single_predict, fg_color=self.colors["button"], hover_color=self.colors["button_hover"])
        predict_btn.grid(row=2, column=0, pady=20, padx=20)

        result_card = ctk.CTkFrame(self.single_view, fg_color=self.colors["card"], corner_radius=12)
        result_card.grid(row=1, column=0, sticky="nsew")
        result_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(result_card, text="Prediction Result", font=self.fonts["card_title"]).pack(padx=20, pady=(15, 10), anchor="w")
        self.single_result_label = ctk.CTkLabel(result_card, text="Result will be displayed here...", font=self.fonts["body"], justify="left", text_color=self.colors["secondary"])
        self.single_result_label.pack(padx=20, pady=10, anchor="w")
        
        self.result_textbox = ctk.CTkTextbox(result_card, wrap="none", font=("Consolas", 12), height=200)
        self.result_textbox.pack(fill="x", expand=True, padx=20, pady=(0, 20))
        self.result_textbox.insert("end", "Detailed site probabilities will be shown here...")
        self.result_textbox.configure(state="disabled")

        # --- Batch Prediction View ---
        self.batch_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.batch_view.grid_columnconfigure(0, weight=1)
        
        batch_input_card = ctk.CTkFrame(self.batch_view, fg_color=self.colors["card"], corner_radius=12)
        batch_input_card.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        batch_input_card.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(batch_input_card, text="Select an Excel file containing sequences for batch processing.", font=self.fonts["body"], text_color=self.colors["secondary"]).grid(row=0, column=0, sticky="w", padx=20, pady=(15, 10))
        
        self.batch_file_entry = ctk.CTkEntry(batch_input_card, placeholder_text="Click 'Select File' to specify the input file path...", height=40, font=self.fonts["body"])
        self.batch_file_entry.grid(row=1, column=0, sticky="ew", padx=20, pady=(0,10))
        self.batch_file_entry.configure(state="disabled")

        button_frame = ctk.CTkFrame(batch_input_card, fg_color="transparent")
        button_frame.grid(row=2, column=0, pady=10, padx=20)
        
        select_file_btn = ctk.CTkButton(button_frame, text="Select File", width=180, height=40, command=self.select_input_file, fg_color=self.colors["button"], hover_color=self.colors["button_hover"])
        select_file_btn.pack(side="left", padx=(0, 10))

        predict_batch_btn = ctk.CTkButton(button_frame, text="Start Batch Prediction", width=180, height=40, command=self.batch_predict, fg_color=self.colors["button"], hover_color=self.colors["button_hover"])
        predict_batch_btn.pack(side="left")

        batch_result_card = ctk.CTkFrame(self.batch_view, fg_color=self.colors["card"], corner_radius=12)
        batch_result_card.grid(row=1, column=0, sticky="nsew")
        
        ctk.CTkLabel(batch_result_card, text="Task Status", font=self.fonts["card_title"]).pack(padx=20, pady=(15, 10), anchor="w")
        self.batch_status_label = ctk.CTkLabel(batch_result_card, text="Please select a file and start the prediction...", font=self.fonts["body"], text_color=self.colors["secondary"])
        self.batch_status_label.pack(padx=20, pady=10, anchor="w")

        # --- Settings View ---
        self.settings_view = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.settings_view.grid_columnconfigure(0, weight=1)

        settings_card = ctk.CTkFrame(self.settings_view, fg_color=self.colors["card"], corner_radius=12)
        settings_card.grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(settings_card, text="Model Settings", font=self.fonts["card_title"]).pack(anchor="w", padx=20, pady=(15,10))
        
        t_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        t_frame.pack(fill="x", padx=20, pady=10)
        self.threshold_label = ctk.CTkLabel(t_frame, text=f"Classification Threshold: {self.threshold:.2f}", font=self.fonts["body"])
        self.threshold_label.pack(side="left")
        ctk.CTkSlider(t_frame, from_=0, to=1, number_of_steps=100, command=self.update_threshold).pack(side="left", padx=15, expand=True, fill="x")

    def create_right_info_panel(self):
        right_panel = ctk.CTkFrame(self, width=300, fg_color=self.colors["card"])
        right_panel.grid(row=0, column=2, sticky="nsw", pady=20, padx=(0, 20))
        right_panel.pack_propagate(False)
        right_panel.grid_rowconfigure(3, weight=1)

        user_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        user_frame.pack(fill="x", pady=20, padx=20)
        ctk.CTkLabel(user_frame, text="👤", font=self.fonts["icon"]).pack(side="left")
        ctk.CTkLabel(user_frame, text="admin", font=self.fonts["body"]).pack(side="left", padx=10)

        ctk.CTkFrame(right_panel, height=1, fg_color=self.colors["nav_hover"]).pack(fill="x", padx=20)
        
        ctk.CTkLabel(right_panel, text="Batch Task Progress", font=self.fonts["card_title"]).pack(anchor="w", pady=(20, 10), padx=20)
        
        self.circular_progress = CircularProgressbar(right_panel, width=150, height=150, progress_width=10, fg_color=self.colors["card"])
        self.circular_progress.pack(pady=10)
        
        ctk.CTkFrame(right_panel, height=1, fg_color=self.colors["nav_hover"]).pack(fill="x", padx=20, pady=(20, 0))
        
        ctk.CTkLabel(right_panel, text="Prediction History", font=self.fonts["card_title"]).pack(anchor="w", pady=(20, 10), padx=20)
        
        self.history_frame = ctk.CTkScrollableFrame(right_panel, fg_color="transparent")
        self.history_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        ctk.CTkLabel(self.history_frame, text="No history yet", text_color=self.colors["secondary"]).pack(pady=10)
    
    def select_view(self, view_name):
        for key, btn in self.nav_buttons.items():
            btn.configure(fg_color="transparent", text_color=self.colors["primary"])

        selected_button = self.nav_buttons.get(view_name)
        if selected_button:
            selected_button.configure(fg_color=self.colors["nav_selected_bg"], text_color=self.colors["nav_selected_fg"])

        if self.current_view:
            self.current_view.grid_forget()

        title_map = {"single": "Single Sequence", "batch": "Batch Prediction", "settings": "Settings"}
        self.view_title.configure(text=title_map.get(view_name, "Dashboard"))

        if view_name == "single":
            self.current_view = self.single_view
        elif view_name == "batch":
            self.current_view = self.batch_view
        elif view_name == "settings":
            self.current_view = self.settings_view
        
        if self.current_view:
            self.current_view.grid(row=1, column=0, sticky="nsew")

    def single_predict(self):
        seq = self.seq_entry.get().strip()
        if not seq:
            messagebox.showwarning("Input Error", "Please enter a sequence!")
            return
        
        try:
            self.set_ui_busy(True)
            result_dict = self.predictor.predict_single(seq, threshold=self.threshold)
            
            
            statistics = result_dict['statistics']
            consistency = result_dict['consistency']
            
            summary = (
                f"Sequence: {result_dict['sequence']}\n"
                f"Predicted Cleavage Sites: {statistics['cleavage_sites_count']} / {statistics['total_positions']}\n"
                f"Average Cleavage Probability: {statistics['avg_probability']:.4f}\n"
                f"Model Consistency (Avg): {consistency['avg_consistency']:.4f}"
            )
            self.single_result_label.configure(text=summary)

            df = pd.DataFrame({
                "Position": range(1, statistics['total_positions'] + 1),
                "Probability": result_dict['probabilities'],
                "Prediction": result_dict['predictions'],
                "Consistency": result_dict['consistency']['consistency_scores']
            })
            df['Probability'] = df['Probability'].map('{:.4f}'.format)
            df['Consistency'] = df['Consistency'].map('{:.4f}'.format)
            
            self.result_textbox.configure(state="normal")
            self.result_textbox.delete("1.0", "end")
            self.result_textbox.insert("end", df.to_string(index=False))
            self.result_textbox.configure(state="disabled")

            history_summary = f"Seq: {result_dict['sequence'][:20]}...\n" \
                              f"Result: {statistics['cleavage_sites_count']} / {statistics['total_positions']} sites"
            self.add_to_history(history_summary)

        except Exception as e:
            messagebox.showerror("Prediction Failed", f"An error occurred during prediction: \n{e}")
        finally:
            self.set_ui_busy(False)

    def add_to_history(self, summary_text):
        if len(self.prediction_history) >= 10:
            self.prediction_history.pop()
        
        self.prediction_history.insert(0, summary_text)

        for widget in self.history_frame.winfo_children():
            widget.destroy()

        for item in self.prediction_history:
            history_item = ctk.CTkFrame(self.history_frame, fg_color=self.colors["nav_hover"], corner_radius=6)
            history_item.pack(fill="x", pady=4, padx=2)
            ctk.CTkLabel(history_item, text=item, font=self.fonts["small"], text_color=self.colors["secondary"], justify="left").pack(padx=8, pady=6, anchor="w")

    def batch_predict(self):
        if self.batch_thread and self.batch_thread.is_alive():
            messagebox.showwarning("Task in Progress", "A batch task is already running. Please wait.")
            return

        input_path = self.batch_input_path
        if not input_path:
            messagebox.showwarning("File Error", "Please select an input file first!")
            return
        
        save_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            initialfile="prediction_results.xlsx",
            filetypes=[("Excel", "*.xlsx")]
        )
        if not save_path:
            self.batch_status_label.configure(text="Task cancelled.", text_color=self.colors["secondary"])
            return

        self.set_ui_busy(True)
        self.batch_status.update({"done": False, "error": None})
        self.circular_progress.start_animation()
        self.batch_status_label.configure(text=f"Processing: {os.path.basename(input_path)}...", text_color=self.colors["secondary"])

        self.batch_thread = threading.Thread(
            target=self._run_batch_thread,
            args=(input_path, save_path),
            daemon=True
        )
        self.batch_thread.start()
        self.after(100, self._check_batch_status)

    def _run_batch_thread(self, input_path, save_path):
        try:
            self.predictor.predict_from_file(
                input_file_path=input_path,
                output_file_path=save_path,
                threshold=self.threshold
            )
        except Exception as e:
            self.batch_status["error"] = e
        finally:
            self.batch_status["done"] = True

    def _check_batch_status(self):
        if self.batch_status["done"]:
            # Stop the animation loop directly
            if self.circular_progress._animation_id:
                self.circular_progress.after_cancel(self.circular_progress._animation_id)
                self.circular_progress._animation_id = None

            self.set_ui_busy(False)
            
            if self.batch_status["error"]:
                error = self.batch_status["error"]
                messagebox.showerror("Batch Prediction Failed", f"An error occurred while processing the file: \n{error}")
                self.batch_status_label.configure(text=f"Processing failed: {error}", text_color="red")
                self.circular_progress.set_progress(0)
            else:
                self.circular_progress.set_progress(100)
                messagebox.showinfo("Complete", "Batch prediction finished!")
                self.batch_status_label.configure(text="Task complete.", text_color=self.colors["secondary"])
        else:
            self.after(100, self._check_batch_status)
            
    def select_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx;*.xls")])
        if file_path:
            self.batch_file_entry.configure(state="normal")
            self.batch_file_entry.delete(0, "end")
            self.batch_file_entry.insert(0, file_path)
            self.batch_file_entry.configure(state="disabled")
            self.batch_input_path = file_path
        else:
            self.batch_file_entry.configure(state="normal")
            self.batch_file_entry.delete(0, "end")
            self.batch_file_entry.configure(state="disabled")
            self.batch_input_path = None

    def update_threshold(self, value):
        self.threshold = float(value)
        self.threshold_label.configure(text=f"Classification Threshold: {self.threshold:.2f}")

    def set_ui_busy(self, is_busy):
        status = "disabled" if is_busy else "normal"
        for btn in self.nav_buttons.values():
            btn.configure(state=status)
        
        for view in [self.single_view, self.batch_view, self.settings_view]:
            for widget in view.winfo_children():
                for sub_widget in widget.winfo_children():
                    if isinstance(sub_widget, ctk.CTkButton):
                        sub_widget.configure(state=status)
                    if isinstance(sub_widget, ctk.CTkFrame):
                        for btn in sub_widget.winfo_children():
                            if isinstance(btn, ctk.CTkButton):
                                btn.configure(state=status)

if __name__ == "__main__":
    app = ModernApp()
    app.mainloop() 

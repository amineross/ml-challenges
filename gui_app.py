import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
from ESPCN import ESPCN
from FSRCNN import FSRCNN
import os
import glob


class UpscaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Upscaler")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Model management
        self.models = {}  # {label: model_path}
        self.current_model = None
        self.device = self.get_device()
        
        # Images
        self.original_image = None
        self.upscaled_image = None
        
        self.setup_ui()
        self.load_available_models()
    
    def get_device(self):
        """Detect best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def setup_ui(self):
        """Create the main UI layout"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top controls
        controls_frame = tk.Frame(main_frame, bg='#2b2b2b')
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Model selection
        model_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(model_frame, text="Model:", bg='#2b2b2b', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)
        
        # Add model button
        add_btn = tk.Button(model_frame, text="Add Model", command=self.add_model, 
                           bg='#4a9eff', fg='white', font=('Arial', 10), relief=tk.FLAT, padx=15)
        add_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Buttons frame
        buttons_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        buttons_frame.pack(side=tk.RIGHT)
        
        # Load image button
        load_btn = tk.Button(buttons_frame, text="Load Image", command=self.load_image,
                            bg='#ff6b4a', fg='white', font=('Arial', 12), relief=tk.FLAT, padx=20)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reprocess button
        self.reprocess_btn = tk.Button(buttons_frame, text="Reprocess", command=self.reprocess_image,
                                      bg='#4a9eff', fg='white', font=('Arial', 12), relief=tk.FLAT, padx=20)
        self.reprocess_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.reprocess_btn.config(state=tk.DISABLED)
        
        # Export button
        self.export_btn = tk.Button(buttons_frame, text="Export", command=self.export_image,
                                   bg='#50c878', fg='white', font=('Arial', 12), relief=tk.FLAT, padx=20)
        self.export_btn.pack(side=tk.LEFT)
        self.export_btn.config(state=tk.DISABLED)
        
        # Image display area
        display_frame = tk.Frame(main_frame, bg='#2b2b2b')
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # Original image
        left_frame = tk.Frame(display_frame, bg='#363636', relief=tk.RAISED, bd=1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(left_frame, text="Original", bg='#363636', fg='white', 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.original_canvas = tk.Canvas(left_frame, bg='#363636', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Upscaled image
        right_frame = tk.Frame(display_frame, bg='#363636', relief=tk.RAISED, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        tk.Label(right_frame, text="Upscaled (4x)", bg='#363636', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.upscaled_canvas = tk.Canvas(right_frame, bg='#363636', highlightthickness=0)
        self.upscaled_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Status bar
        self.status_var = tk.StringVar(value=f"Ready - Device: {self.device}")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, bg='#1e1e1e', fg='#888',
                             font=('Arial', 9), anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def load_available_models(self):
        """Load models from artifacts directory"""
        model_files = glob.glob("artifacts/*.pt")
        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            self.models[model_name] = model_path
        
        self.update_model_combo()
    
    def update_model_combo(self):
        """Update the model combobox with available models"""
        model_names = list(self.models.keys())
        self.model_combo['values'] = model_names
        if model_names and not self.model_var.get():
            self.model_var.set(model_names[0])
            self.load_model(self.models[model_names[0]])
    
    def add_model(self):
        """Add a new model with custom label"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if file_path:
            # Get custom label
            dialog = ModelLabelDialog(self.root)
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                label = dialog.result
                self.models[label] = file_path
                self.update_model_combo()
                self.model_var.set(label)
                self.load_model(file_path)
    
    def on_model_select(self, event=None):
        """Handle model selection"""
        selected = self.model_var.get()
        if selected in self.models:
            self.load_model(self.models[selected])
    
    def load_model(self, model_path):
        """Load the selected model"""
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Validate checkpoint structure
            required_keys = ["model_state_dict", "model_class"]
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"Missing required key in checkpoint: {key}")
            
            # Create model instance based on checkpoint data
            if checkpoint["model_class"] == "ESPCN":
                # Default to 4 if not saved
                self.current_model = ESPCN()
            elif checkpoint["model_class"] == "FSRCNN":
                # Get FSRCNN parameters from checkpoint or use default
                self.current_model = FSRCNN()
            else:
                raise ValueError(f"Unknown model class: {checkpoint['model_class']}")
            
            self.current_model.load_state_dict(checkpoint["model_state_dict"])
            self.current_model.to(self.device)
            self.current_model.eval()
            
            epochs = checkpoint.get("epochs", "unknown")
            model_class = checkpoint["model_class"]
            scale = checkpoint.get("scale", "unknown")
            self.status_var.set(f"{model_class} model loaded (scale={scale}) - {epochs} epochs - Device: {self.device}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.status_var.set(f"Error loading model - Device: {self.device}")
    
    def load_image(self):
        """Load an image for upscaling"""
        if not self.current_model:
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("Processing image...")
                self.root.update()
                
                # Load and display original
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image, self.original_canvas)
                
                # Upscale
                self.upscaled_image = self.upscale_image(self.original_image)
                self.display_image(self.upscaled_image, self.upscaled_canvas)
                
                # Enable buttons after successful processing
                self.reprocess_btn.config(state=tk.NORMAL)
                self.export_btn.config(state=tk.NORMAL)
                
                self.status_var.set(f"Image processed - {self.original_image.size} → {self.upscaled_image.size}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process image: {str(e)}")
                self.status_var.set(f"Error processing image - Device: {self.device}")
    
    def upscale_image(self, image):
        """Upscale image using the current model"""
        # Convert to tensor
        img_array = np.array(image.convert('RGB'))
        img_tensor = torch.from_numpy(img_array).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            upscaled_tensor = self.current_model(img_tensor)
        
        # Convert back to PIL
        upscaled_array = upscaled_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        upscaled_array = np.clip(upscaled_array * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(upscaled_array)
    
    def display_image(self, image, canvas):
        """Display image on canvas with proper scaling"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready, schedule for later
            self.root.after(100, lambda: self.display_image(image, canvas))
            return
        
        # Calculate scaling to fit canvas
        img_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width - 20
            new_height = int(new_width / img_ratio)
        else:
            new_height = canvas_height - 20
            new_width = int(new_height * img_ratio)
        
        # Resize and display
        display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        canvas.image = photo  # Keep reference
    
    def reprocess_image(self):
        """Reprocess the current image with the selected model"""
        if not self.current_model:
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        if not self.original_image:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        try:
            self.status_var.set("Reprocessing image...")
            self.root.update()
            
            # Upscale with current model
            self.upscaled_image = self.upscale_image(self.original_image)
            self.display_image(self.upscaled_image, self.upscaled_canvas)
            
            self.status_var.set(f"Image reprocessed - {self.original_image.size} → {self.upscaled_image.size}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reprocess image: {str(e)}")
            self.status_var.set(f"Error reprocessing image - Device: {self.device}")
    
    def export_image(self):
        """Export the upscaled image"""
        if not self.upscaled_image:
            messagebox.showwarning("Warning", "No upscaled image to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Upscaled Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.upscaled_image.save(file_path)
                self.status_var.set(f"Image exported to: {file_path}")
                messagebox.showinfo("Success", "Image exported successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export image: {str(e)}")
                self.status_var.set(f"Error exporting image - Device: {self.device}")


class ModelLabelDialog:
    def __init__(self, parent):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Model Label")
        self.dialog.geometry("300x120")
        self.dialog.configure(bg='#2b2b2b')
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="Enter model label:", bg='#2b2b2b', fg='white',
                font=('Arial', 11)).pack(pady=15)
        
        self.entry = tk.Entry(self.dialog, font=('Arial', 11), width=25)
        self.entry.pack(pady=5)
        self.entry.focus()
        self.entry.bind('<Return>', self.on_ok)
        
        btn_frame = tk.Frame(self.dialog, bg='#2b2b2b')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="OK", command=self.on_ok, bg='#4a9eff', fg='white',
                 font=('Arial', 10), relief=tk.FLAT, padx=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Cancel", command=self.on_cancel, bg='#666', fg='white',
                 font=('Arial', 10), relief=tk.FLAT, padx=15).pack(side=tk.LEFT, padx=5)
    
    def on_ok(self, event=None):
        self.result = self.entry.get().strip()
        if self.result:
            self.dialog.destroy()
    
    def on_cancel(self):
        self.dialog.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = UpscaleApp(root)
    root.mainloop() 
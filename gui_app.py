import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
from ESPCN import ESPCN
from FSRCNN import FSRCNN
from EDSRLITE import EDSRLITE
import os
import glob


class UpscaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("upscaler d'images")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # gestion des modèles
        self.models = {}  # {label: model_path}
        self.current_model = None
        self.device = self.get_device()
        
        # images
        self.original_image = None
        self.upscaled_image = None
        
        self.setup_ui()
        self.load_available_models()
    
    def get_device(self):
        """détecter le meilleur périphérique disponible"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def setup_ui(self):
        """créer l'interface utilisateur principale"""
        # conteneur principal
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # contrôles du haut
        controls_frame = tk.Frame(main_frame, bg='#2b2b2b')
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # sélection du modèle
        model_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        model_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Label(model_frame, text="modèle:", bg='#2b2b2b', fg='white', font=('Arial', 12)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=30)
        self.model_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_select)
        
        # bouton ajouter modèle
        add_btn = tk.Button(model_frame, text="ajouter modèle", command=self.add_model, 
                           bg='#4a9eff', fg='white', font=('Arial', 10), relief=tk.FLAT, padx=15)
        add_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # frame des boutons
        buttons_frame = tk.Frame(controls_frame, bg='#2b2b2b')
        buttons_frame.pack(side=tk.RIGHT)
        
        # bouton charger image
        load_btn = tk.Button(buttons_frame, text="charger image", command=self.load_image,
                            bg='#ff6b4a', fg='white', font=('Arial', 12), relief=tk.FLAT, padx=20)
        load_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # bouton reprocesser
        self.reprocess_btn = tk.Button(buttons_frame, text="reprocesser", command=self.reprocess_image,
                                      bg='#4a9eff', fg='white', font=('Arial', 12), relief=tk.FLAT, padx=20)
        self.reprocess_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.reprocess_btn.config(state=tk.DISABLED)
        
        # bouton exporter
        self.export_btn = tk.Button(buttons_frame, text="exporter", command=self.export_image,
                                   bg='#50c878', fg='white', font=('Arial', 12), relief=tk.FLAT, padx=20)
        self.export_btn.pack(side=tk.LEFT)
        self.export_btn.config(state=tk.DISABLED)
        
        # zone d'affichage des images
        display_frame = tk.Frame(main_frame, bg='#2b2b2b')
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        # image originale
        left_frame = tk.Frame(display_frame, bg='#363636', relief=tk.RAISED, bd=1)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        tk.Label(left_frame, text="originale", bg='#363636', fg='white', 
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.original_canvas = tk.Canvas(left_frame, bg='#363636', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # image upscalée
        right_frame = tk.Frame(display_frame, bg='#363636', relief=tk.RAISED, bd=1)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        tk.Label(right_frame, text="upscalée (4x)", bg='#363636', fg='white',
                font=('Arial', 14, 'bold')).pack(pady=10)
        
        self.upscaled_canvas = tk.Canvas(right_frame, bg='#363636', highlightthickness=0)
        self.upscaled_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # barre de statut
        self.status_var = tk.StringVar(value=f"prêt - périphérique: {self.device}")
        status_bar = tk.Label(main_frame, textvariable=self.status_var, bg='#1e1e1e', fg='#888',
                             font=('Arial', 9), anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def load_available_models(self):
        """charger les modèles depuis le répertoire artifacts"""
        model_files = glob.glob("artifacts/*.pt")
        for model_path in model_files:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            self.models[model_name] = model_path
        
        self.update_model_combo()
    
    def update_model_combo(self):
        """mettre à jour la combobox des modèles avec les modèles disponibles"""
        model_names = list(self.models.keys())
        self.model_combo['values'] = model_names
        if model_names and not self.model_var.get():
            self.model_var.set(model_names[0])
            self.load_model(self.models[model_names[0]])
    
    def add_model(self):
        """ajouter un nouveau modèle avec un label personnalisé"""
        file_path = filedialog.askopenfilename(
            title="sélectionner fichier modèle",
            filetypes=[("fichiers pytorch", "*.pt"), ("tous fichiers", "*.*")]
        )
        
        if file_path:
            # obtenir label personnalisé
            dialog = ModelLabelDialog(self.root)
            self.root.wait_window(dialog.dialog)
            
            if dialog.result:
                label = dialog.result
                self.models[label] = file_path
                self.update_model_combo()
                self.model_var.set(label)
                self.load_model(file_path)
    
    def on_model_select(self, event=None):
        """gérer la sélection de modèle"""
        selected = self.model_var.get()
        if selected in self.models:
            self.load_model(self.models[selected])
    
    def load_model(self, model_path):
        """charger le modèle sélectionné"""
        try:
            self.status_var.set("chargement du modèle...")
            self.root.update()
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # valider la structure du checkpoint
            required_keys = ["model_state_dict", "model_class"]
            for key in required_keys:
                if key not in checkpoint:
                    raise KeyError(f"clé requise manquante dans le checkpoint: {key}")
            
            # créer l'instance du modèle basée sur les données du checkpoint
            if checkpoint["model_class"] == "ESPCN":
                # par défaut 4 si pas sauvegardé
                self.current_model = ESPCN()
            elif checkpoint["model_class"] == "FSRCNN":
                # obtenir les paramètres fsrcnn du checkpoint ou utiliser par défaut
                self.current_model = FSRCNN()
            elif checkpoint["model_class"] == "EDSRLITE":
                # obtenir les paramètres edsrlite du checkpoint ou utiliser par défaut
                self.current_model = EDSRLITE()
            else:
                raise ValueError(f"classe de modèle inconnue: {checkpoint['model_class']}")
            
            self.current_model.load_state_dict(checkpoint["model_state_dict"])
            self.current_model.to(self.device)
            self.current_model.eval()
            
            epochs = checkpoint.get("epochs", "inconnu")
            model_class = checkpoint["model_class"]
            scale = checkpoint.get("scale", "inconnu")
            self.status_var.set(f"modèle {model_class} chargé (échelle={scale}) - {epochs} epochs - périphérique: {self.device}")
            
        except Exception as e:
            messagebox.showerror("erreur", f"échec du chargement du modèle: {str(e)}")
            self.status_var.set(f"erreur chargement modèle - périphérique: {self.device}")
    
    def load_image(self):
        """charger une image pour l'upscaling"""
        if not self.current_model:
            messagebox.showwarning("attention", "veuillez d'abord sélectionner un modèle")
            return
        
        file_path = filedialog.askopenfilename(
            title="sélectionner image",
            filetypes=[("fichiers image", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("tous fichiers", "*.*")]
        )
        
        if file_path:
            try:
                self.status_var.set("traitement de l'image...")
                self.root.update()
                
                # charger et afficher l'originale
                self.original_image = Image.open(file_path)
                self.display_image(self.original_image, self.original_canvas)
                
                # upscaler
                self.upscaled_image = self.upscale_image(self.original_image)
                self.display_image(self.upscaled_image, self.upscaled_canvas)
                
                # activer les boutons après traitement réussi
                self.reprocess_btn.config(state=tk.NORMAL)
                self.export_btn.config(state=tk.NORMAL)
                
                self.status_var.set(f"image traitée - {self.original_image.size} → {self.upscaled_image.size}")
                
            except Exception as e:
                messagebox.showerror("erreur", f"échec du traitement de l'image: {str(e)}")
                self.status_var.set(f"erreur traitement image - périphérique: {self.device}")
    
    def upscale_image(self, image):
        """upscaler l'image avec le modèle actuel"""
        # convertir en tensor (éviter from_numpy, passer par dlpack)
        img_array = np.array(image.convert('RGB'))
        t = torch.utils.dlpack.from_dlpack(img_array.__dlpack__())  # [H,W,3] uint8 cpu
        img_tensor = t.permute(2, 0, 1).to(dtype=torch.float32).div(255.0)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # inférence
        with torch.no_grad():
            upscaled_tensor = self.current_model(img_tensor)
        
        # reconvertir en pil sans .numpy()
        x = (
            upscaled_tensor.squeeze(0)
            .permute(1, 2, 0)
            .contiguous()
            .to("cpu")
        )
        upscaled_array = np.from_dlpack(x)
        upscaled_array = np.clip(upscaled_array * 255.0, 0, 255).astype(np.uint8)
        
        return Image.fromarray(upscaled_array)
    
    def display_image(self, image, canvas):
        """afficher l'image sur le canvas avec mise à l'échelle appropriée"""
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # canvas pas prêt, programmer pour plus tard
            self.root.after(100, lambda: self.display_image(image, canvas))
            return
        
        # calculer la mise à l'échelle pour s'adapter au canvas
        img_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width - 20
            new_height = int(new_width / img_ratio)
        else:
            new_height = canvas_height - 20
            new_width = int(new_height * img_ratio)
        
        # redimensionner et afficher
        display_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display_image)
        
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, image=photo)
        canvas.image = photo  # garder la référence
    
    def reprocess_image(self):
        """reprocesser l'image actuelle avec le modèle sélectionné"""
        if not self.current_model:
            messagebox.showwarning("attention", "veuillez d'abord sélectionner un modèle")
            return
        
        if not self.original_image:
            messagebox.showwarning("attention", "veuillez d'abord charger une image")
            return
        
        try:
            self.status_var.set("retraitement de l'image...")
            self.root.update()
            
            # upscaler avec le modèle actuel
            self.upscaled_image = self.upscale_image(self.original_image)
            self.display_image(self.upscaled_image, self.upscaled_canvas)
            
            self.status_var.set(f"image retraitée - {self.original_image.size} → {self.upscaled_image.size}")
            
        except Exception as e:
            messagebox.showerror("erreur", f"échec du retraitement de l'image: {str(e)}")
            self.status_var.set(f"erreur retraitement image - périphérique: {self.device}")
    
    def export_image(self):
        """exporter l'image upscalée"""
        if not self.upscaled_image:
            messagebox.showwarning("attention", "aucune image upscalée à exporter")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="sauvegarder image upscalée",
            defaultextension=".png",
            filetypes=[("fichiers png", "*.png"), ("fichiers jpeg", "*.jpg"), ("tous fichiers", "*.*")]
        )
        
        if file_path:
            try:
                self.upscaled_image.save(file_path)
                self.status_var.set(f"image exportée vers: {file_path}")
                messagebox.showinfo("succès", "image exportée avec succès!")
                
            except Exception as e:
                messagebox.showerror("erreur", f"échec de l'export de l'image: {str(e)}")
                self.status_var.set(f"erreur export image - périphérique: {self.device}")


class ModelLabelDialog:
    def __init__(self, parent):
        self.result = None
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("label du modèle")
        self.dialog.geometry("300x120")
        self.dialog.configure(bg='#2b2b2b')
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # centrer la boîte de dialogue
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        tk.Label(self.dialog, text="entrer le label du modèle:", bg='#2b2b2b', fg='white',
                font=('Arial', 11)).pack(pady=15)
        
        self.entry = tk.Entry(self.dialog, font=('Arial', 11), width=25)
        self.entry.pack(pady=5)
        self.entry.focus()
        self.entry.bind('<Return>', self.on_ok)
        
        btn_frame = tk.Frame(self.dialog, bg='#2b2b2b')
        btn_frame.pack(pady=15)
        
        tk.Button(btn_frame, text="ok", command=self.on_ok, bg='#4a9eff', fg='white',
                 font=('Arial', 10), relief=tk.FLAT, padx=15).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="annuler", command=self.on_cancel, bg='#666', fg='white',
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
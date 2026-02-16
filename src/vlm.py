"""LLaVA Vision-Language Model integration for clinical description generation."""

import torch
from PIL import Image
import numpy as np
from typing import Optional, Dict
from pathlib import Path


class LlavaGenerator:
    """
    LLaVA-based clinical description generator for polyp analysis.
    
    This class integrates a Vision-Language Model to generate clinical descriptions
    of detected polyps based on the original image and segmentation mask.
    """
    
    def __init__(self, model_name: str = "llava-hf/llava-1.5-7b-hf", device: str = "cuda"):
        """
        Initialize LLaVA model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"\n[INFO] Loading LLaVA model: {model_name}")
        print(f"   Device: {self.device}")
        
        try:
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            
            self.processor = LlavaNextProcessor.from_pretrained(model_name)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            
            print("[OK] LLaVA model loaded successfully")
            
        except ImportError:
            print("[WARN] Warning: transformers library not installed or LLaVA model not available")
            print("   Install with: pip install transformers accelerate")
            self.model = None
            self.processor = None
    
    def create_overlay(
        self,
        image_path: str,
        mask_path: str,
        output_path: Optional[str] = None
    ) -> Image.Image:
        """
        Create an overlay of the segmentation mask on the original image.
        
        Args:
            image_path: Path to original image
            mask_path: Path to segmentation mask
            output_path: Optional path to save overlay
            
        Returns:
            PIL Image with mask overlay
        """
        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize mask to match image if needed
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.NEAREST)
        
        # Convert to numpy
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Create colored overlay (green for polyp region)
        overlay = image_np.copy()
        overlay[mask_np > 127] = overlay[mask_np > 127] * 0.6 + np.array([0, 255, 0]) * 0.4
        
        # Convert back to PIL
        overlay_img = Image.fromarray(overlay.astype(np.uint8))
        
        if output_path:
            overlay_img.save(output_path)
        
        return overlay_img
    
    def generate_description(
        self,
        image_path: str,
        mask_path: str,
        prompt_template: Optional[str] = None
    ) -> str:
        """
        Generate clinical description of polyp using LLaVA.
        
        Args:
            image_path: Path to original colonoscopy image
            mask_path: Path to segmentation mask
            prompt_template: Custom prompt template (optional)
            
        Returns:
            Generated clinical description
        """
        if self.model is None or self.processor is None:
            return self._generate_fallback_description(image_path, mask_path)
        
        # Create overlay for better context
        overlay = self.create_overlay(image_path, mask_path)
        
        # Default clinical prompt
        if prompt_template is None:
            prompt_template = """[INST] <image>
You are an expert gastroenterologist analyzing a colonoscopy image with a highlighted polyp region (shown in green overlay).

Please provide a detailed clinical assessment including:
1. Polyp size estimation (small <5mm, medium 5-10mm, large >10mm)
2. Polyp shape and morphology (sessile, pedunculated, flat)
3. Surface characteristics (smooth, irregular, ulcerated)
4. Location and distribution
5. Recommended follow-up actions based on findings

Provide your assessment in a professional clinical format. [/INST]"""
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt_template,
            images=overlay,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        print("\n[INFO] Generating clinical description...")
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        description = self.processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the response part (after [/INST])
        if "[/INST]" in description:
            description = description.split("[/INST]")[-1].strip()
        
        return description
    
    def _generate_fallback_description(self, image_path: str, mask_path: str) -> str:
        """
        Generate a basic description when LLaVA is not available.
        
        This analyzes the mask to provide basic metrics.
        """
        import cv2
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Calculate basic metrics
        total_pixels = mask.shape[0] * mask.shape[1]
        polyp_pixels = np.sum(binary_mask > 0)
        polyp_percentage = (polyp_pixels / total_pixels) * 100
        
        # Find contours for size estimation
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return "No polyp detected in the segmentation mask."
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Size estimation (rough)
        avg_dimension = (w + h) / 2
        if avg_dimension < 50:
            size_category = "small (<5mm)"
        elif avg_dimension < 100:
            size_category = "medium (5-10mm)"
        else:
            size_category = "large (>10mm)"
        
        # Generate description
        description = f"""CLINICAL ASSESSMENT (Automated Analysis):

1. POLYP DETECTION:
   - Status: Polyp detected and segmented
   - Coverage: {polyp_percentage:.2f}% of image area
   
2. SIZE ESTIMATION:
   - Bounding box: {w}x{h} pixels
   - Estimated size category: {size_category}
   
3. MORPHOLOGY:
   - Number of detected regions: {len(contours)}
   - Primary region dimensions: {w}x{h} pixels
   
4. RECOMMENDATIONS:
   - Manual review by gastroenterologist required
   - Consider histopathological examination
   - Follow standard surveillance protocols based on size and morphology

NOTE: This is an automated preliminary assessment. 
Clinical decisions should be made by qualified medical professionals 
after thorough examination of the complete colonoscopy footage.
"""
        return description
    
    def batch_generate(
        self,
        image_mask_pairs: list,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Generate descriptions for multiple image-mask pairs.
        
        Args:
            image_mask_pairs: List of (image_path, mask_path) tuples
            output_dir: Directory to save descriptions
            
        Returns:
            Dictionary mapping image names to descriptions
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        descriptions = {}
        
        print(f"\n[INFO] Generating descriptions for {len(image_mask_pairs)} images...")
        
        for i, (img_path, mask_path) in enumerate(image_mask_pairs, 1):
            print(f"\nProcessing {i}/{len(image_mask_pairs)}: {Path(img_path).name}")
            
            try:
                description = self.generate_description(img_path, mask_path)
                
                # Save description
                img_name = Path(img_path).stem
                desc_file = output_path / f"{img_name}_description.txt"
                
                with open(desc_file, 'w') as f:
                    f.write(f"Image: {img_path}\n")
                    f.write(f"Mask: {mask_path}\n")
                    f.write(f"\n{'='*80}\n")
                    f.write(description)
                
                descriptions[img_name] = description
                print(f"[OK] Description saved to {desc_file}")
                
            except Exception as e:
                print(f"[WARN] Error processing {img_path}: {e}")
                descriptions[Path(img_path).stem] = f"Error: {str(e)}"
        
        print(f"\n[OK] Batch processing complete. {len(descriptions)} descriptions generated.")
        return descriptions


def test_llava_installation() -> bool:
    """Test if LLaVA dependencies are properly installed."""
    try:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        print("[OK] LLaVA dependencies are installed")
        return True
    except ImportError as e:
        print(f"[FAIL] LLaVA dependencies missing: {e}")
        print("\nTo install, run:")
        print("  pip install transformers accelerate pillow torch")
        return False

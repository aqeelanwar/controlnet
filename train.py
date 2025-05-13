import os
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel, DDPMScheduler
from diffusers import AutoencoderKL, StableDiffusionPipeline
from transformers import AutoTokenizer, CLIPTextModel
import torch
from accelerate import Accelerator

class SchemaToSEMDataset(Dataset):
    def __init__(self, schema_dir, sem_dir, prompt_file, image_size=512):
        self.schema_dir = schema_dir
        self.sem_dir = sem_dir
        self.prompts = json.load(open(prompt_file))
        self.filenames = list(self.prompts.keys())
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        fname = self.filenames[idx]
        schema_path = os.path.join(self.schema_dir, fname)
        sem_path = os.path.join(self.sem_dir, fname)

        schema_img = self.transform(Image.open(schema_path).convert("RGB"))
        sem_img = self.transform(Image.open(sem_path).convert("RGB"))
        prompt = self.prompts[fname]
        return {"schema": schema_img, "sem": sem_img, "prompt": prompt}

def collate_fn(batch, tokenizer):
    prompts = [item['prompt'] for item in batch]
    input_ids = tokenizer(prompts, padding="max_length", truncation=True, return_tensors="pt").input_ids
    schema_images = torch.stack([item['schema'] for item in batch])
    sem_images = torch.stack([item['sem'] for item in batch])
    return {
        "input_ids": input_ids,
        "schema_images": schema_images,
        "sem_images": sem_images
    }

def train_controlnet():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    base_model = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    text_encoder = CLIPTextModel.from_pretrained(base_model).to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device)
    
    # Load ControlNet and wrap pipeline
    controlnet = ControlNetModel.from_unet(unet)
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        safety_checker=None,
        revision="fp16",
        torch_dtype=torch.float16
    ).to(device)

    # Dataset
    dataset = SchemaToSEMDataset(
        schema_dir="data/train/schema",
        sem_dir="data/train/sem",
        prompt_file="data/train/prompts.json",
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer)
    )

    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=1e-5)
    accelerator = Accelerator()
    controlnet, optimizer, dataloader = accelerator.prepare(controlnet, optimizer, dataloader)

    # Training Loop
    controlnet.train()
    for epoch in range(10):
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(controlnet):
                encoder_hidden_states = text_encoder(batch["input_ids"].to(device))[0]
                noise_pred = controlnet(
                    sample=batch["sem_images"].to(device),
                    timestep=torch.randint(0, 1000, (batch["sem_images"].shape[0],), device=device).long(),
                    encoder_hidden_states=encoder_hidden_states,
                    control=batch["schema_images"].to(device),
                ).sample

                loss = torch.nn.functional.mse_loss(noise_pred, batch["sem_images"].to(device))
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    # Save model
    accelerator.unwrap_model(controlnet).save_pretrained("controlnet-trained")
    tokenizer.save_pretrained("controlnet-trained")

if __name__ == "__main__":
    train_controlnet()

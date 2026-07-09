import random
import cv2
import einops
import numpy as np
from pytorch_lightning import seed_everything
import torch
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from medical_dataset import MedicalDataset, MedicalDatasetGenerate
from PIL import Image
import os, argparse
from annotator.util import resize_image

parser = argparse.ArgumentParser()
parser.add_argument('--attr_type', type=str, help='attribute type')
#parser.add_argument('--name', type=str, help='specific attribute to train')
parser.add_argument("--ckpt", type=str)
parser.add_argument("--images", type=int, default=10000)
parser.add_argument("--save_path", type=str, default="trash/TEST/")

opt = parser.parse_args()

assert opt.ckpt is not None
assert os.path.exists(opt.ckpt)
os.makedirs(opt.save_path, exist_ok=True)
os.makedirs(opt.save_path+"/hint", exist_ok=True)
os.makedirs(opt.save_path+"/sample", exist_ok=True)
os.makedirs(opt.save_path+"/full", exist_ok=True)


if __name__ == "__main__":
    
    seed = 0
    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)
    
    if opt.attr_type=='ISIC':
        syn_label_dir = '/path/to/your/DATASET/ISIC_resize/masks/all/train'
        pt = "High-resolution dermoscopic image of a malignant melanoma on human skin. The lesion has asymmetric shape, irregular borders, multiple colors (dark brown, black, blue-gray), and a blue-white veil. ISIC dataset style, clinical quality, macro shot, dermatology dataset."
    elif opt.attr_type=='Polyp':
        pt = "A high-resolution colonoscopy image showing the inside of the colon with realistic lighting and texture. A small to medium-sized polyp is clearly visible on the colon wall, with natural folds and moist surfaces. Medical imaging style, endoscopic view, clean and clinical appearance."
        syn_label_dir = '/path/to/your/DATASET/Polyp_resize/masks/all/train'
    elif opt.attr_type=='Nuclei':
        syn_label_dir = '/path/to/your/DATASET/Nuclei_resize/masks/all/train'
        pt = "Breast pathology slide showing: Invasive carcinoma, H&E staining, 20x magnification, Clear nuclear pleomorphism"
    elif opt.attr_type=='Prostate':
        syn_label_dir = '/path/to/your/DATASET/Prostate_resize/masks/all/train'
        pt = "Generate a high-resolution T2-weighted MRI image of human prostate"
    elif opt.attr_type=='Fundus':
        syn_label_dir = '/path/to/your/DATASET/Fundus_resize/masks/all/train'
        #pt = "Realistic fundus image, human retina, optic disc and blood vessels visible, medical imaging style."
        #pt="Realistic fundus image, human retina. Clearly visible optic disc containing a paler, central optic cup. Blood vessels visible, medical imaging style."
        pt="Realistic fundus image, human retina. The optic disc is visible as a bright oval structure. Within the disc, the optic cup appears as a distinct, pale, central depression. Retinal blood vessels converge and bend over the rim of the cup. Medical imaging style."
    elif opt.attr_type=='Chaos':
        syn_label_dir = '/path/to/your/DATASET/Chaos_resize/masks/all/train'
        pt = "Clear 2D CT slice showing: - Liver/Spleen/Kidneys - Abdominal aorta - Psoas muscles, DICOM style, diagnostic quality"
    elif opt.attr_type=='FeTS':
        syn_label_dir = '/path/to/your/DATASET/FeTS_resize/masks/all/train'
        pt = "Brain MRI image showing: High-grade glioma, T2-weighted/FLAIR, Hyperintense lesion with peritumoral edema, Irregular margins, Necrotic core"
    dataset = MedicalDatasetGenerate(syn_label_dir, pt)
    
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(opt.ckpt, location='cpu'))
    model = model.cuda()
    
    ddim_sampler = DDIMSampler(model)
    
    num_samples = 1
    ddim_steps = 20
    scale = 9      #9
    eta = 0.0
    strength = 1.0
    prompt = pt


    a_prompt = "low quality, blurry"
    n_prompt = "lowres,extra digit, fewer digits, cropped, worst quality"
    
    with torch.no_grad():
        
        for i in range(opt.images):
            term = dataset[i % len(dataset)]
    
            control = torch.stack([torch.tensor(term['hint']).cuda() for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            
            H, W = control.shape[2:]

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)
            print(shape)
            model.control_scales = ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)
            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            Image.fromarray(x_samples[0]).save(f"{opt.save_path}/sample/{i:06d}.png")
            
            # save control
            Image.fromarray((term['hint'] * 255.0).astype(np.uint8)).save(f"{opt.save_path}/hint/{i:06d}.png")
            
            # # save original
            # Image.fromarray((term['jpg'] * 127.5 + 127.5).cpu().numpy().astype(np.uint8)).save(f"{opt.save_path}/original_{i:06d}.png")

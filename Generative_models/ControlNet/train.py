from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from medical_dataset import MedicalDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from datetime import datetime
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--attr_type', type=str, help='attribute type')
    parser.add_argument('--epochs', type=int, help='save frequency')
    parser.add_argument('--gpu', type=int, help='specific gpu to train')
    args = parser.parse_args()

    # Configs
    resume_path = './models/control_sd15_seg.pth'
    batch_size = 1
    logger_freq = 300
    learning_rate = 1e-5
    sd_locked = True
    # sd_locked = False
    only_mid_control = True
    save_frequency = args.epochs


    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control
    checkpoint_callback = ModelCheckpoint(monitor=None, every_n_epochs=save_frequency, save_top_k=-1, filename='{epoch:02d}', save_last=True)


    # Misc
    img_files = ''
    pt = ""
    if args.attr_type=='ISIC':
        img_files = '/path/to/your/DATASET/ISIC_resize/imgs/train'
        pt = "High-resolution dermoscopic image of a malignant melanoma on human skin. The lesion has asymmetric shape, irregular borders, multiple colors (dark brown, black, blue-gray), and a blue-white veil. ISIC dataset style, clinical quality, macro shot, dermatology dataset."
    elif args.attr_type=='Chaos':
        img_files = '/path/to/your/DATASET/Chaos_resize/imgs/train'
        pt = "Clear 2D CT slice showing: - Liver/Spleen/Kidneys - Abdominal aorta - Psoas muscles, DICOM style, diagnostic quality"
    elif args.attr_type=='FeTS':
        img_files = '/path/to/your/DATASET/FeTS_resize/imgs/train'
        pt = "Brain MRI image showing: High-grade glioma, T2-weighted/FLAIR, Hyperintense lesion with peritumoral edema, Irregular margins, Necrotic core"
    elif args.attr_type=='Polyp':
        img_files = '/path/to/your/DATASET/Polyp_resize/imgs/train'
        pt = "Pathology slide showing: Invasive carcinoma, H&E staining, 20x magnification, Clear nuclear pleomorphism"
    elif args.attr_type=='Nuclei':
        img_files = '/path/to/your/DATASET/Nuclei_resize/imgs/train'
        pt = "Pathology slide showing: Invasive carcinoma, H&E staining, 20x magnification, Clear nuclear pleomorphism"
    elif args.attr_type=='Prostate':
        img_files = '/path/to/your/DATASET/Prostate_resize/imgs/train'
        pt = "Generate a high-resolution T2-weighted MRI image of human prostate"
    elif args.attr_type=='Fundus':
        img_files = '/path/to/your/DATASET/Fundus_resize/imgs/train'
        pt="Realistic fundus image, human retina. The optic disc is visible as a bright oval structure. Within the disc, the optic cup appears as a distinct, pale, central depression. Retinal blood vessels converge and bend over the rim of the cup. Medical imaging style."
    attribute = args.attr_type
    files = '/DATA_EDS/datasets/MedicalImage/txt/' + attribute + '.txt'
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = f"log_{current_time}_{args.attr_type}"

    dataset = MedicalDataset(img_files, pt)
    print(img_files)
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(gpus=[args.gpu], precision=32, callbacks=[logger, checkpoint_callback], default_root_dir=log_dir)


    # Train!
    trainer.fit(model, dataloader)

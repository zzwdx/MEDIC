from torchvision import transforms
from torch.utils.data import DataLoader
from dataset.dataset import SingleDomainData, MultiDomainData

def get_transform(instr, small_img=False, color_jitter=True, random_grayscale=True):
    if small_img == False:
        img_tr = [transforms.RandomResizedCrop((224, 224), (0.8, 1.0))]
        img_tr.append(transforms.RandomHorizontalFlip(0.5))
        if color_jitter:
            img_tr.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4))
        if random_grayscale:
            img_tr.append(transforms.RandomGrayscale(0.1))
        img_tr.append(transforms.ToTensor())
        img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform_map = {     
            "train": transforms.Compose(img_tr),
            "val": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            "test": transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
    else:
        img_tr = [transforms.Resize((32, 32))]
        if color_jitter:
            img_tr.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4))
        if random_grayscale:
            img_tr.append(transforms.RandomGrayscale(0.1))
        img_tr.append(transforms.ToTensor())
        img_tr.append(transforms.Normalize([0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        transform_map = {     
            "train": transforms.Compose(img_tr),
            "val": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]),
            "test": transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        }

    return transform_map[instr]

def get_dataloader(root_dir, domain, classes, batch_size, domain_class_dict=None, get_domain_label=True, get_class_label=True, instr="train", small_img=False, shuffle=True, drop_last=True, num_workers=4):
    if isinstance(domain, list): 
        dataset = MultiDomainData(root_dir=root_dir, domain=domain, classes=classes, domain_class_dict=domain_class_dict, get_domain_label=get_domain_label, get_classes_label=get_class_label, transform=get_transform(instr, small_img=small_img))
    else:
        dataset = SingleDomainData(root_dir=root_dir, domain=domain, classes=classes, domain_label=-1, get_classes_label=get_class_label, transform=get_transform(instr, small_img=small_img))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader



    









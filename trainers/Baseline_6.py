import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from libraries import *
from utils import *
from models.Baseline_6 import Baseline_6

Configs = yaml.safe_load(open("Configs.yaml"))
Environment = yaml.safe_load(open("Environment.yaml"))

def prepare_dataset(videos_root, tracking_root, working_root,
                    train_data = [], val_data = []):
    train_image_label = {}
    val_image_label = {}
    
    videos = os.listdir(videos_root)
    for video in tqdm(videos, desc="Processing Videos"):
        video_path = os.path.join(videos_root, video)
        if not os.path.isdir(video_path): continue

        video_annot = os.path.join(video_path, 'annotations.txt')

        clip_category_dct = load_video_annot(video_annot)
        
        clips = os.listdir(video_path)
        for clip in clips:
            
            clip_path = os.path.join(video_path, clip)
            if not os.path.isdir(clip_path): continue
                
            clip_annot = os.path.join(tracking_root, video, clip, f'{clip}.txt')
            
            frame_boxes = load_tracking_annot(clip_annot)

            items = list(frame_boxes.items())
            key, value = items[5]  
            label = Configs['data']['categories_dct'][clip_category_dct[f'{key}']]

            clip_output_path = os.path.join(working_root, video, clip)
            os.makedirs(clip_output_path, exist_ok = True)
            
            for frame_id, boxes_info in frame_boxes.items():
                img_path = os.path.join(clip_path, f'{frame_id}.jpg')

                if frame_id == key : players_sorted, right_padding = sorting(img_path, boxes_info)
                image = Image.open(img_path).convert('RGB')

                frame_output_path = os.path.join(clip_output_path, f'{frame_id}')
                os.makedirs(frame_output_path, exist_ok=True)
    
                for i, box_info in enumerate(boxes_info):
                    cropped_image = image.crop(box_info.box)
                    resized_image = cropped_image.resize((224, 224), Image.BILINEAR)
            
                    save_name = f"f{frame_id}_p{box_info.player_ID}.jpg"
                    path = os.path.join(frame_output_path, save_name)
                    
                    resized_image.save(path)
            if video in train_data : train_image_label[clip_output_path] = (label, players_sorted, right_padding)
            elif video in val_data : val_image_label[clip_output_path] = (label, players_sorted, right_padding)
                    
    with open(f'{working_root}/train_image_labels.pkl', 'wb') as f:
        pickle.dump(train_image_label, f)

    with open(f'{working_root}/val_image_labels.pkl', 'wb') as f :
        pickle.dump(val_image_label, f)

class dataset(Dataset):
    def __init__(self, data = [], data_type = '', working_root = ''):
        if data_type == 'train' : 
            self.transform = transforms.Compose([
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=15, sigma=(5.0, 5.0))
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else : 
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.working_root = working_root
        self.players_paths = []  

        videos = data
        for video in videos:
            video_path = os.path.join(self.working_root, video)
            
            if os.path.isdir(video_path):
                clips_ids = os.listdir(video_path)
                
                for clip_id in clips_ids:
                    clip_id_path = os.path.join(video_path, clip_id)

                    self.players_paths.append(clip_id_path)
                        

        with open(f'{working_root}/{data_type}_image_labels.pkl', 'rb') as f:
            self.image_label = pickle.load(f)

    def __len__(self):
        return len(self.players_paths)
                
    def __getitem__(self, idx):
        players_path = self.players_paths[idx]
        frames_ids = sorted(os.listdir(players_path), key=int)
        label, players_sorted, right_padding = self.image_label[self.working_root + "/" + "/".join(players_path.split("/")[-2:])] # integer
        
        clip_images = []
        clip_mask = []
        for frame_id in frames_ids:    
            frame_id_path = os.path.join(players_path, frame_id)

            frame_images = []
            for i in players_sorted :
                    player_id_path = os.path.join(frame_id_path, f"f{frame_id}_p{i}.jpg")
                    image = Image.open(player_id_path)
                    processed_image = self.transform(image) 
                    frame_images.append(processed_image) # [tensor.size([3, 224, 224]), tensor.size(....), .....] len() --> 12
                
            frame_images = torch.stack(frame_images, dim = 0) # tensor.size([12, 3, 224, 224])
            
            mask = torch.cat(
                (torch.ones(frame_images.size(0), dtype=torch.bool),torch.zeros(12 - frame_images.size(0), dtype=torch.bool)), dim=0
            ) # tensor.size([12])
            
            players_number = frame_images.size(0)
            if players_number != 12 :
                clip_padding = torch.zeros((12 - players_number, 3, 224, 224))
                
                if right_padding :
                    frame_images = torch.cat((frame_images, clip_padding), dim=0) # tensor.size([12, 3, 224, 224])
                else :
                    frame_images = torch.cat((clip_padding, frame_images), dim=0) # tensor.size([12, 3, 224, 224])
 
            clip_images.append(frame_images) # [tensor.size([12, 3, 224, 224])...]->len(9)
            clip_mask.append(mask) # [tensor.size([12]), ....]->len(9)
        clip_images = torch.stack(clip_images, dim = 0) # tensor.size([9, 12, 3, 224, 224])
        clip_mask = torch.stack(clip_mask, dim = 0) # tensor.size([9,12])
        
        return clip_images, label, clip_mask


if __name__ == '__main__':
    
    prepare_dataset(
      Environment['videos_root'],
      Environment['videos_tracking_annot'],
      Environment['working_root'],
      train_data=Configs['data']['train_ids'],
      val_data=Configs['data']['test_ids']
    )
    
    train_dataset = dataset(
        data = Configs['data']['train_ids'],
        data_type = 'train',
        working_root = Environment['working_root']
    )
          
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Configs['Baseline_6']['model']['batch_size'], 
        shuffle=True,
        num_workers=4
    )
      
    val_dataset = dataset(      
        data = Configs['data']['test_ids'],
        data_type = 'val',
        working_root = Environment['working_root']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Configs['Baseline_6']['model']['batch_size'], 
        shuffle=False, 
        num_workers=4,
    )
    
    device = torch.device(Configs['device'])
    model = Baseline_6()
    tuned_weights = torch.load(Environment['tuned_weights_path'], map_location = device)
    name_map = {'conv1': '0', 'bn1': '1', 'layer1': '4', 'layer2': '5', 'layer3': '6', 'layer4': '7'}   
    remapped_weights = {}
    for k, v in tuned_weights['model_state_dict'].items():
        if not k.startswith('backbone.'):
            continue
        parts = k.split('.')
        if parts[1] in name_map:
            parts[1] = name_map[parts[1]]
            remapped_weights['.'.join(parts)] = v
    
    model.load_state_dict(remapped_weights, strict=False)

    model = nn.DataParallel(model)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda') 

    optimizer = optim.AdamW(
      model.parameters(),
      lr = Configs['Baseline_6']['model']['lr'],
      weight_decay = Configs['Baseline_6']['model']['weight_decay']
    )	

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )
    
    graph_train_losses = []
    graph_val_losses = []
    for epoch in range(Configs['Baseline_6']['num_epochs']):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        train_labels_list = []
        train_preds_list = []
        for images, labels, mask in train_loader:
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                t_outputs = model(images, mask)
                t_loss = criterion(t_outputs, labels)
    
            scaler.scale(t_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_num_in_batch = labels.numel()
            train_loss += t_loss.item() * t_num_in_batch
            train_samples+= t_num_in_batch

            _, train_preds = torch.max(t_outputs, 1)
            train_correct += (train_preds == labels).sum().item()

            train_preds_list.append(train_preds.cpu().tolist())
            train_labels_list.append(labels.cpu().tolist())

        train_preds_list = np.concatenate(train_preds_list)
        train_labels_list = np.concatenate(train_labels_list)

        final_train_loss = train_loss / train_samples
        final_train_acc = (train_correct / train_samples) * 100


        model.eval() 
        val_loss = 0.0
        val_correct = 0
        val_samples = 0
        val_preds = []
        val_labels_list = []

        with torch.no_grad(): 
            for images, labels, mask in val_loader:
                images, labels, mask = images.to(device), labels.to(device), mask.to(device)
                
                with torch.amp.autocast('cuda'):
                    v_outputs = model(images, mask)
                    v_loss = criterion(v_outputs, labels)
    
                num_in_batch = labels.numel()
                val_loss += v_loss.item() * num_in_batch
                val_samples += num_in_batch
                
                _, preds = torch.max(v_outputs, 1)
                val_correct += (preds == labels).sum().item()
                
                val_preds.append(preds.cpu().tolist())
                val_labels_list.append(labels.cpu().tolist())

        val_preds = np.concatenate(val_preds)
        val_labels_list = np.concatenate(val_labels_list)
        
        final_val_loss = val_loss / val_samples
        final_val_acc = (val_correct / val_samples) * 100

        train_cm = confusion_matrix(train_labels_list, train_preds_list)
        val_cm = confusion_matrix(val_labels_list, val_preds)

        f1_train = f1_score(train_labels_list, train_preds_list, average='weighted')
        f1_val = f1_score(val_labels_list, val_preds, average='weighted')


        print('-------------------------------------------')
        print(f"\n--- train Results (Epoch {epoch+1}) ---")
        print(f"train Loss: {final_train_loss:.4f} - train Acc: {final_train_acc:.2f}% - train_f1_weighted: {f1_train:.2f}")
        print(f"\n--- Validation Results (Epoch {epoch+1}) ---")
        print(f"Val Loss: {final_val_loss:.4f} - Val Acc: {final_val_acc:.2f}% - val_f1_weighted: {f1_val:.2f}")

        plt.figure(figsize=(6,5))
        sns.heatmap(train_cm, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=Configs['data']['group_classes']  , yticklabels=Configs['data']['group_classes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Train Confusion Matrix')
        plt.show()

        plt.figure(figsize=(6,5))
        sns.heatmap(val_cm, annot=True, fmt='d', cmap='coolwarm',
                    xticklabels=Configs['data']['group_classes'], yticklabels=Configs['data']['group_classes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Val Confusion Matrix')
        plt.show()
        
        graph_train_losses.append(final_train_loss)
        graph_val_losses.append(final_val_loss)
        epochs = range(1, len(graph_train_losses) + 1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, graph_train_losses, marker='o', label='Train Loss')
        plt.plot(epochs, graph_val_losses, marker='o', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        scheduler.step(final_val_loss)
        torch.cuda.empty_cache()
        gc.collect()

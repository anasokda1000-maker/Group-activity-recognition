from libraries import *
from utils import *
from model import person_classifier

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

        clips = os.listdir(video_path)
        for clip in clips:
            
            clip_path = os.path.join(video_path, clip)
            if not os.path.isdir(clip_path): continue
                
            clip_annot = os.path.join(tracking_root, video, clip, f'{clip}.txt')
            
            frame_boxes = load_tracking_annot(clip_annot)

            for frame_id, boxes_info in frame_boxes.items():
                img_path = os.path.join(clip_path, f'{frame_id}.jpg')
                image = Image.open(img_path).convert('RGB')
                    
                frame_output_path = os.path.join(working_root, video, clip, f'{frame_id}')
                os.makedirs(frame_output_path, exist_ok=True)

                for i, box_info in enumerate(boxes_info):
                    cropped_image = image.crop(box_info.box)
                    resized_image = cropped_image.resize((224, 224), Image.BILINEAR)
                    label = Configs['data']['player_label_dct'][box_info.category]
            
                    save_name = f"f{frame_id}_p{i}.jpg"
                    path = os.path.join(frame_output_path, save_name)
                    
                    resized_image.save(path)
                    if video in train_data : train_image_label[path] = label
                    elif video in val_data : val_image_label[path] = label
                    
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
        self.players_paths = []  

        videos = data
        for video in videos:
            video_path = os.path.join(working_root, video)
            
            if os.path.isdir(video_path):
                clips_ids = os.listdir(video_path)
                
                for clip_id in clips_ids:
                    clip_id_path = os.path.join(video_path, clip_id)

                    if os.path.isdir(clip_id_path):
                        frames_id = os.listdir(clip_id_path)
                                              
                        for frame_id in frames_id :
                            frame_id_path = os.path.join(clip_id_path, frame_id)
        
                            if os.path.isdir(frame_id_path):
                                save_names = os.listdir(frame_id_path)
                                
                                for save_name in save_names :
                                    save_name_path = os.path.join(frame_id_path, save_name)
                                    self.players_paths.append(save_name_path)
                
        pkl_path = f'{working_root}/{data_type}_image_labels.pkl'
        
        with open(pkl_path, 'rb') as f:
            self.image_label = pickle.load(f)

    def __len__(self):
        return len(self.players_paths)
                
    def __getitem__(self, idx):
        player_path = self.players_paths[idx]
                
        with Image.open(player_path) as image:
            processed_image = self.transform(image) # tensor.size([3, 224, 224])
        label = self.image_label[player_path] # integer
        
        return processed_image, label 

if __name__ == '__main__':
    
	prepare_dataset(
        Environment['videos_root'],
        Environment['videos_tracking_annot'],
        Environment['working_root'],
        train_data=Configs['data']['train_ids'],
        val_data=Configs['data']['val_ids']
    )
    
	train_dataset = dataset(
        data = Configs['data']['train_ids'],
        data_type = 'train',
        working_root = Environment['working_root']
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Configs['fine_tuning']['model']['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    val_dataset = dataset(      
        data = Configs['data']['val_ids'],
        data_type = 'val',
        working_root = Environment['working_root']
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Configs['fine_tuning']['model']['batch_size'], 
        shuffle=False, 
        num_workers=4,
    )

    device = torch.device(Configs['device'])
	model = person_classifier()
    model = nn.DataParallel(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler('cuda')  

    optimizer = optim.AdamW(
      model.parameters(),
      lr = Configs['fine_tuning']['model']['lr'],
      weight_decay = Configs['fine_tuning']['model']['weight_decay']
    )	
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5
    ) 

    best_weights_path = f'{Environment['working_root']}/model/best_weights_pth'
    os.makedirs(best_weights_path, exist_ok = True)
	best_val_acc = 0.0
    for epoch in range(Configs['fine_tuning']['num_epochs']):

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_samples = 0
        train_labels_list = []
        train_preds_list = []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                t_outputs = model(images)
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
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda'):
                    v_outputs = model(images)
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
        train_cm_percent = (train_cm/ train_cm.sum(axis=1, keepdims=True) ) * 100
        
        val_cm = confusion_matrix(val_labels_list, val_preds)
        val_cm_percent = (val_cm / val_cm.sum(axis=1, keepdims=True) ) * 100

        f1_weighted = f1_score(val_labels_list, val_preds, average='weighted')
        
        print(f"\n--- train Results (Epoch {epoch+1}) ---")
        print(f"train Loss: {final_train_loss:.4f} - train Acc: {final_train_acc:.2f}%")
        print('-------------------------------------------')
        print(f"\n--- Validation Results (Epoch {epoch+1}) ---")
        print(f"Val Loss: {final_val_loss:.4f} - Val Acc: {final_val_acc:.2f}% - f1 weighted: {f1_weighted:.4f}")

        plt.figure(figsize=(6,5))
        sns.heatmap(train_cm_percent, annot=True, fmt='.1f', cmap='coolwarm',
                    xticklabels= Configs['data']['player_classes'], yticklabels= Configs['data']['player_classes'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Train Confusion Matrix')
        plt.show()
        
        plt.figure(figsize=(6,5))
        sns.heatmap(val_cm_percent, annot=True, fmt=".1f", cmap="coolwarm",
                    xticklabels= Configs['data']['player_classes'], yticklabels= Configs['data']['player_classes'])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Val Confusion Matrix")
        plt.show()

        if final_val_acc > best_val_acc:
            print(f" Improved: Val Acc increased from {best_val_acc:.2f}% to {final_val_acc:.2f}%")
            best_val_acc = final_val_acc
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  
                'scaler_state_dict': scaler.state_dict(),        
            }, best_weights_path)
            print(f" Saved best model weights to: {best_weights_path}")
        else:
            print(f" No improvement in Val Acc. (Current best: {best_val_acc:.2f}%)")
        

        scheduler.step(final_val_loss)
        torch.cuda.empty_cache()
        gc.collect()
        
print(f"\nTraining Finished! Best Validation Accuracy achieved: {best_val_acc:.2f}%")

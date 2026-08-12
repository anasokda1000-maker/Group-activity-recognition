from PIL import Image

class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated
      

def load_tracking_annot(path):
  with open(path, 'r') as file:
      player_boxes = {idx:[] for idx in range(12)}
      frame_boxes_dct = {}

      for idx, line in enumerate(file):
          box_info = BoxInfo(line)
          if box_info.player_ID > 11:
              continue
          player_boxes[box_info.player_ID].append(box_info)

      # let's create view from frame to boxes
      for player_ID, boxes_info in player_boxes.items():
          # let's keep the middle 9 frames only (enough for this task empirically)
          boxes_info = boxes_info[5:]
          boxes_info = boxes_info[:-6]

          for box_info in boxes_info:
              if box_info.frame_ID not in frame_boxes_dct:
                  frame_boxes_dct[box_info.frame_ID] = []

              frame_boxes_dct[box_info.frame_ID].append(box_info)

      return frame_boxes_dct

def load_video_annot(video_annot):
    with open(video_annot, 'r') as file:
        clip_category_dct = {}

        for line in file:
            items = line.strip().split(' ')[:2]
            clip_dir = items[0].replace('.jpg', '')
            clip_category_dct[clip_dir] = items[1]

        return clip_category_dct

def sorting(image_path, boxes_info):
  image = Image.open(image_path)
  width, _ = image.size
  lowest_x = float('inf')
  highest_x = 0.0
  x1_sorting = []
  for box_info in boxes_info:
      x1, _, x2, _ = box_info.box
      x1_sorting.append((x1, box_info.player_ID))
      if x1 < lowest_x : lowest_x = x1
      if x2 > highest_x: highest_x= x2

  x1_sorted = sorted(x1_sorting, key=lambda x: x[0])
  players_sorted = [player_id for _, player_id in x1_sorted]
  if lowest_x > width - highest_x : right_padding = True
  else : right_padding = False

  return players_sorted, right_padding

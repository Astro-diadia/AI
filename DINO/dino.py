from groundingdino.util.inference import Model
import torch
import OCR

# py "D:\dino.py"

class DINO:
    def __init__(self):
        self.config = "C:\\Users\\lolex\\AppData\\Local\\Programs\\Python\\Python311\\Lib\\site-packages\\groundingdino\\config\\GroundingDINO_SwinB_cfg.py"
        self.ckpt = "D:\Models\groundingdino_swinb_cogcoor.pth"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = Model(
                model_config_path=self.config,
                model_checkpoint_path=self.ckpt,
                device=self.device
            )

    def detect_once(self, image, prompts):
        output = {}
        for prompt in prompts.split(","):
            prompt = prompt.strip()
            box, score = self.model.predict_with_caption(
                image=image,
                caption=prompt,
                box_threshold=0.35,
                text_threshold=0.25
            )
            if box is None or len(box.xyxy) == 0:
                output[prompt] = f"there is no {prompt}"
            else:
                output[prompt] = box.xyxy.tolist()[0]
        return output
        

# for box in boxes:
#     x1, y1, x2, y2 = box[0]
#     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#     cv2.imshow("Detection", image)
#     cv2.waitKey(0)

# cv2.destroyAllWindows()

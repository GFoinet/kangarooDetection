import torch
from PIL import Image, ImageDraw
from torchvision import transforms as T
import glob

device = torch.device('cuda')
model = torch.load('test.pt')
model.to(device)
model.eval()

transformer = T.Compose([
    T.ToTensor(),
    ])

imgList = glob.glob('test/*.jpg')
for imgFilename in imgList:
    with Image.open(imgFilename).convert("RGB") as img:
        imgt = transformer(img).unsqueeze(0)
        pred = model(imgt.to(device))
        print(pred)

        # draw
        base = img.convert("RGBA")
        txt = Image.new("RGBA", base.size, (255,255,255,0))
        # get a drawing context
        d = ImageDraw.Draw(txt)
        for bbox, score in zip(pred[0]['boxes'].tolist(),pred[0]['scores'].tolist()):
            if score > 0.5:
                d.rectangle(bbox,fill=None,width = 10)
        
        out = Image.alpha_composite(base, txt)
        out.show()

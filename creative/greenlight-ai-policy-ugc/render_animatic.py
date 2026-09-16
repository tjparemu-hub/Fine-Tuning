import math, random, os
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import imageio_ffmpeg

W,H,FPS,DUR = 1080,1920,24,30.0
F="/mnt/skills/examples/canvas-design/canvas-fonts/"
def fnt(n,s): return ImageFont.truetype(F+n,s)
CAP   = fnt("WorkSans-Bold.ttf",64)
CAPs  = fnt("WorkSans-Bold.ttf",52)
OSTF  = fnt("WorkSans-Bold.ttf",46)
MONO  = fnt("JetBrainsMono-Bold.ttf",24)
MONOs = fnt("JetBrainsMono-Regular.ttf",20)

BG=(9,12,11); INK=(237,239,234); GO=(95,191,137); STOP=(233,99,90); CHECK=(215,164,69)
ROOM=(20,26,25); ROOM2=(27,34,32); FIG=(96,110,105); LINE=(48,56,54)

BEATS=[
 dict(i=1,a=0.0,b=2.5,t="th",size="MCU",vo="Two Copilots on your laptop. Only one's actually ours.",
      ost="TWO COPILOTS",lanes=[("GREENLIT",GO),("HARD STOP",STOP)]),
 dict(i=2,a=2.5,b=5.5,t="screen",size="INSERT",vo="Same icon, same name, completely different rules.",
      ost=None,lanes=[("HARD STOP",STOP)]),
 dict(i=3,a=5.5,b=9.5,t="th",size="MCU",vo="Work runs inside our Microsoft tenant. That's the safe one.",
      ost="WORK = INSIDE OUR TENANT",ostc=GO,lanes=[("GREENLIT",GO)]),
 dict(i=4,a=9.5,b=14.0,t="hands",size="INSERT",
      vo="Internal, confidential, client-confidential, restricted — personal data too. All fine in Work.",
      ost="ALL FINE IN WORK",ostc=GO,lanes=[("GREENLIT",GO)]),
 dict(i=5,a=14.0,b=17.5,t="point",size="MCU",vo="Web is the public internet. Nothing of ours goes in.",
      ost="WEB = PUBLIC",ostc=STOP,lanes=[("HARD STOP",STOP)]),
 dict(i=6,a=17.5,b=24.0,t="count",size="MCU",
      vo="Three catches: privileged stuff stays out of Copilot entirely. A contract that bans AI beats your licence. And no recording externals without everyone's yes.",
      ost=None,lanes=[("CHECK FIRST",CHECK)]),
 dict(i=7,a=24.0,b=27.0,t="th",size="MCU",vo="Either way, it still gets a human read before it ships.",
      ost=None,lanes=[("CHECK FIRST",CHECK)]),
 dict(i=8,a=27.0,b=30.0,t="grab",size="OTS",vo="Check the toggle before you type.",
      ost="CHECK THE TOGGLE",sub="licence? helpdesk@nteractive.com",lanes=[("GREENLIT",GO)]),
]
HOT={"work":GO,"web":STOP}

# ---------- storyboard plates ----------
def room(d):
    d.rectangle([60,150,430,420],fill=ROOM2)                      # whiteboard
    for y in range(200,400,46): d.line([90,y,380,y],fill=(45,53,51),width=3)
    d.rectangle([700,230,1010,450],outline=LINE,width=3)          # window
    d.line([855,230,855,450],fill=LINE,width=2); d.line([700,340,1010,340],fill=LINE,width=2)
    d.line([0,1345,W,1345],fill=LINE,width=3)                     # desk line
    d.rectangle([740,1160,1010,1345],fill=ROOM)                   # monitor
    for x in range(120,260,26): d.line([x,1345,x-30,1500],fill=(38,45,43),width=4)  # cables

def person(d,cx,cy,arm=None,sc=1.0):
    r=int(150*sc)
    d.polygon([(cx-int(300*sc),cy+int(720*sc)),(cx-int(250*sc),cy+int(300*sc)),
               (cx-int(120*sc),cy+int(180*sc)),(cx+int(120*sc),cy+int(180*sc)),
               (cx+int(250*sc),cy+int(300*sc)),(cx+int(300*sc),cy+int(720*sc))],fill=FIG)
    d.ellipse([cx-r,cy-r-20,cx+r,cy+r-20],fill=FIG)               # head
    d.ellipse([cx-int(70*sc),cy-int(150*sc),cx+int(70*sc),cy-int(115*sc)],fill=(64,74,71))  # clip
    g=int(58*sc); gy=cy-int(30*sc)
    d.rounded_rectangle([cx-int(118*sc),gy-g//2,cx-int(18*sc),gy+g//2],8,outline=BG,width=6)
    d.rounded_rectangle([cx+int(18*sc),gy-g//2,cx+int(118*sc),gy+g//2],8,outline=BG,width=6)
    d.line([cx-int(18*sc),gy,cx+int(18*sc),gy],fill=BG,width=6)
    if arm=="point":
        d.polygon([(cx+int(180*sc),cy+int(620*sc)),(cx+int(300*sc),cy+int(430*sc)),
                   (cx+int(390*sc),cy+int(470*sc)),(cx+int(280*sc),cy+int(680*sc))],fill=FIG)
        d.ellipse([cx+int(340*sc),cy+int(400*sc),cx+int(470*sc),cy+int(520*sc)],fill=(90,103,99))
    if arm=="count":
        bx,by=cx+int(210*sc),cy+int(420*sc)
        for k,hh in enumerate((150,185,150)):
            d.rounded_rectangle([bx+k*54,by-hh,bx+k*54+40,by+70],20,fill=(90,103,99))

def plate(kind):
    im=Image.new("RGB",(W,H),BG); d=ImageDraw.Draw(im)
    if kind in ("th","point","count"):
        room(d); person(d,int(W*0.42),760,arm=("point" if kind=="point" else ("count" if kind=="count" else None)))
    elif kind=="screen":
        room(d)
        d.rounded_rectangle([110,620,970,1290],14,fill=(43,52,49))
        d.rectangle([160,670,920,1150],fill=(21,26,25))
        d.rectangle([200,710,510,790],fill=(62,74,70)); d.rectangle([540,710,850,790],fill=(42,51,48))
        for y in range(850,1120,42): d.rectangle([200,y,200+random.randint(300,640),y+14],fill=(35,42,40))
        d.ellipse([560,660,700,860],fill=(90,103,99))
    elif kind=="hands":
        room(d)
        d.rounded_rectangle([40,880,1040,1320],12,fill=(43,52,49))
        d.rectangle([110,930,970,1180],fill=(21,26,25))
        for r_ in range(4):
            for c in range(11):
                d.rounded_rectangle([130+c*76,1200+r_*28,190+c*76,1220+r_*28],5,fill=(52,61,58))
        d.ellipse([190,1180,470,1420],fill=(90,103,99)); d.ellipse([610,1180,890,1420],fill=(90,103,99))
        d.rectangle([260,330,820,720],fill=(24,31,30))
    elif kind=="grab":
        base=Image.new("RGB",(W,H),BG); bd=ImageDraw.Draw(base)
        room(bd); person(bd,int(W*0.44),760)
        bd.ellipse([560,760,1080,1400],fill=(96,110,105))
        base=base.rotate(-8,resample=Image.BICUBIC,fillcolor=BG).filter(ImageFilter.GaussianBlur(11))
        im=base
    # vignette + window key light
    v=Image.new("L",(W,H),0); vd=ImageDraw.Draw(v)
    vd.ellipse([-330,-620,W+330,H+620],fill=255); v=v.filter(ImageFilter.GaussianBlur(220))
    im=Image.composite(im,Image.new("RGB",(W,H),(4,6,6)),v)
    key=Image.new("L",(W,H),0); kd=ImageDraw.Draw(key)
    kd.ellipse([-620,-120,760,1500],fill=34); key=key.filter(ImageFilter.GaussianBlur(260))
    im=ImageChops.add(im,Image.merge("RGB",(key,key.point(lambda p:int(p*.97)),key.point(lambda p:int(p*.88)))))
    return im

PLATES={k:plate(k) for k in ("th","point","count","screen","hands","grab")}
random.seed(7)
GRAIN=[]
for _ in range(6):
    g=Image.effect_noise((W//3,H//3),22).resize((W,H),Image.BILINEAR).point(lambda p:int(p*0.17))
    GRAIN.append(Image.merge("RGB",(g,g,g)))

# ---------- caption groups ----------
def groups(vo,a,b):
    ws=vo.split(); out=[]; i=0
    while i<len(ws):
        n=4 if len(ws)-i>=5 else len(ws)-i
        out.append(ws[i:i+n]); i+=n
    tot=len(ws); t=a; res=[]
    for g in out:
        dt=(b-a)*len(g)/tot; res.append((t,t+dt,g)); t+=dt
    return res
for B in BEATS: B["groups"]=groups(B["vo"],B["a"],B["b"])

def wrap(draw,words,f,maxw):
    lines=[[]];
    for w in words:
        trial=" ".join(lines[-1]+[w])
        if draw.textlength(trial,font=f)>maxw and lines[-1]: lines.append([w])
        else: lines[-1].append(w)
    return lines

def clean(w): return w.strip(".,:—?!").lower()

def draw_caption(d,words,shown,y0):
    f = CAP if len(" ".join(words))<=30 else CAPs
    lines=wrap(d,words,f,900); idx=0
    lh=int(f.size*1.22); total=len(lines)*lh; y=y0-total
    for ln in lines:
        wtot=sum(d.textlength(w+" ",font=f) for w in ln)-d.textlength(" ",font=f)
        x=(W-wtot)/2
        pad=16
        d.rounded_rectangle([x-pad,y-10,x+wtot+pad,y+f.size+18],10,fill=(10,13,12))
        for w in ln:
            on = idx<shown
            col=HOT.get(clean(w),INK) if on else (60,68,65)
            d.text((x+2,y+3),w,font=f,fill=(0,0,0))
            d.text((x,y),w,font=f,fill=col if on else (48,55,53))
            x+=d.textlength(w+" ",font=f); idx+=1
        y+=lh

def tc(t):
    return "00:%04.1f"%t

# ---------- render ----------
out="/tmp/claude-0/-home-user-Fine-Tuning/9246c648-154a-5e95-b0a0-0c020584cbec/scratchpad/ep01.mp4"
wr=imageio_ffmpeg.write_frames(out,(W,H),fps=FPS,quality=7,macro_block_size=8,
    output_params=["-pix_fmt","yuv420p","-movflags","+faststart"])
wr.send(None)
N=int(DUR*FPS)
for n in range(N):
    t=n/FPS
    B=[b for b in BEATS if b["a"]<=t][-1]
    loc=(t-B["a"])
    dx=int(9*math.sin(t*1.9+B["i"])+5*math.sin(t*4.3)); dy=int(7*math.cos(t*1.5+B["i"])+4*math.sin(t*3.1))
    zoom=1.0+0.012*loc
    pl=PLATES[B["t"]]
    zw,zh=int(W*zoom),int(H*zoom)
    im=pl.resize((zw,zh),Image.BILINEAR).crop(((zw-W)//2-dx,(zh-H)//2-dy,(zw-W)//2-dx+W,(zh-H)//2-dy+H))
    if loc<0.10: im=im.filter(ImageFilter.GaussianBlur(7*(1-loc/0.10)))   # focus hunt
    im=ImageChops.add(im,GRAIN[n%6])
    if t<0.25: im=Image.blend(Image.new("RGB",(W,H),(0,0,0)),im,t/0.25)
    if t>29.75: im=Image.blend(im,Image.new("RGB",(W,H),(0,0,0)),(t-29.75)/0.25)
    d=ImageDraw.Draw(im,"RGBA")
    # lane chips
    x=56
    for lab,col in B["lanes"]:
        tw=d.textlength(lab,font=MONO)
        d.rounded_rectangle([x,64,x+tw+32,64+46],4,fill=col)
        d.text((x+16,74),lab,font=MONO,fill=(8,10,10)); x+=tw+44
    # corner data
    d.text((W-56-d.textlength(B["size"],font=MONO),74),B["size"],font=MONO,fill=(150,160,155))
    d.text((56,H-140),"GREENLIGHT  EP01  TWO COPILOTS",font=MONOs,fill=(110,120,116))
    d.text((56,H-108),tc(t)+"  SHOT %d/8"%B["i"],font=MONOs,fill=(110,120,116))
    lab="PREVIZ — NOT FINAL FOOTAGE"
    d.text((W-56-d.textlength(lab,font=MONOs),H-108),lab,font=MONOs,fill=(110,120,116))
    # OST card
    if B.get("ost"):
        s=min(1.0,loc/0.25); oy=300
        tw=d.textlength(B["ost"],font=OSTF)
        col=B.get("ostc",INK)
        d.rounded_rectangle([(W-tw)/2-26,oy-16,(W+tw)/2+26,oy+OSTF.size+22],6,fill=col+(int(255*s),))
        d.text(((W-tw)/2,oy),B["ost"],font=OSTF,fill=(10,13,12,int(255*s)))
        if B.get("sub"):
            sw=d.textlength(B["sub"],font=MONO)
            d.text(((W-sw)/2,oy+OSTF.size+42),B["sub"],font=MONO,fill=(200,208,203,int(255*s)))
    # captions
    for (ga,gb,gw) in B["groups"]:
        if ga<=t<gb:
            shown=min(len(gw),int((t-ga)/max(.001,(gb-ga))*len(gw))+1)
            draw_caption(d,gw,shown,1470); break
    # progress hairline
    d.rectangle([0,H-6,int(W*t/DUR),H],fill=GO)
    wr.send(im.tobytes())
wr.close()
print("done",os.path.getsize(out))

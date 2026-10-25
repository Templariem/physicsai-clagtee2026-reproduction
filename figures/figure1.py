"""Reposition the four original verification panels in a 2x2 layout, without replotting."""
from pathlib import Path
from pypdf import PdfReader,PdfWriter,PageObject,Transformation
from copy import deepcopy
R=Path(__file__).resolve().parents[1];src=PdfReader(R/'fem/results/P1/fig1_verification.pdf').pages[0]
w,h=float(src.mediabox.width),float(src.mediabox.height);tile=w/4
out=PageObject.create_blank_page(width=tile*2,height=h*2)
for i in range(4):
 p=deepcopy(src);p.cropbox.lower_left=(tile*i,0);p.cropbox.upper_right=(tile*(i+1),h)
 out.merge_transformed_page(p,Transformation().translate(tx=(i%2)*tile-i*tile,ty=(1-i//2)*h))
dest=R/'runs/figures';dest.mkdir(parents=True,exist_ok=True);writer=PdfWriter();writer.add_page(out);writer.write(dest/'figure1.pdf')

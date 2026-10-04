## Limitations
- The auditor checks **category** consistency only, not brand or attributes. Tested example:
  a Seiko watch photo with the title "Fastrack Men Analog Watch" and category Watches is marked OK.
  Brand verification (logo recognition, OCR on the product, or image–text matching models like CLIP)
  would be needed to catch this.

## FINDINGS
- ~2/3 of the image model's 127 disagreements come from two look-alike pairs:
  Shoes ↔ Sandal (61) and Topwear ↔ Innerwear (25). The asymmetry (40 shoes predicted as
  sandals vs 21 the other way) is consistent with sandals being filed under Shoes in the catalog.
## Data & attribution
Product images and metadata: *Fashion Product Images (Small)* by Param Aggarwal on Kaggle
(released under the MIT License). Product photos originally from Myntra.
This project is a non-commercial portfolio demonstration.
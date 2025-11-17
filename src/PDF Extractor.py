import pytesseract
from pdf2image import convert_from_path
textdatalist = []
for item in os.listdir("path"):
    if str(item) == '.DS_Store':
        pass
    else:
        pages = convert_from_path("path" + str(item), 600)
        text_data = ''
        for page in pages:
            text = pytesseract.image_to_string(page)
            text_data += text + '\n'
        textdatalist.append(text_data)

date_list = []
type_list = []
patent_list = []
figure_list = []
for my_text in textdatalist:
    for i in range(0,len(my_text)):
        if my_text[i:i+8] == 'Pub. No.':
            patent_list.append(my_text[i+6:i+28])
        if my_text[i:i+10] == 'Patent No.':
            patent_list.append(my_text[i+8:i+29])

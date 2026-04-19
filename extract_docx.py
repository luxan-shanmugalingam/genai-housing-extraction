import zipfile
from xml.etree.cElementTree import XML

WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PARA = WORD_NAMESPACE + 'p'
TEXT = WORD_NAMESPACE + 't'

try:
    with zipfile.ZipFile(r'G:\My Drive\University academic matters\7th semester\ST 4035 Data Science\Group project\Housing rental Prices app\Group E Data Science Project Report.docx') as zf:
        xml_content = zf.read('word/document.xml')
    tree = XML(xml_content)
    paragraphs = []
    for paragraph in tree.iter(PARA):
        texts = [node.text for node in paragraph.iter(TEXT) if node.text]
        if texts:
            paragraphs.append(''.join(texts))
    with open('report_text.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(paragraphs))
    print("Extraction successful.")
except Exception as e:
    print(f"Error: {e}")

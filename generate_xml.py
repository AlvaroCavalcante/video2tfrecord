import xml.etree.cElementTree as ET


class AnnotationGenerator(object):
    def __init__(self, xml_path):
        self.xml_path = xml_path

    def generate_xml_annotation(self, output_bbox, im_width, im_height, file_name):
        try:
            annotation = ET.Element('annotation')
            ET.SubElement(annotation, 'filename').text = file_name
            size = ET.SubElement(annotation, 'size')
            ET.SubElement(size, 'width').text = str(im_width)
            ET.SubElement(size, 'height').text = str(im_height)
            ET.SubElement(size, 'depth').text = '3'
            
            count = 0
            for class_name in output_bbox:
                box = output_bbox.get(class_name)

                objectBox = ET.SubElement(annotation, 'object')
                ET.SubElement(objectBox, 'name').text = class_name
                ET.SubElement(objectBox, 'pose').text = 'Unspecified'
                ET.SubElement(objectBox, 'truncated').text = '0'
                ET.SubElement(objectBox, 'difficult').text = '0'
                bndBox = ET.SubElement(objectBox, 'bndbox')
                ET.SubElement(bndBox, 'xmin').text = str(box['xmin'])
                ET.SubElement(bndBox, 'ymin').text = str(box['ymin'])
                ET.SubElement(bndBox, 'xmax').text = str(box['xmax'])
                ET.SubElement(bndBox, 'ymax').text = str(box['ymax'])
                count += 1

            arquivo = ET.ElementTree(annotation)
            arquivo.write(self.xml_path + file_name + '.xml')
        except Exception as e:
            print('Error to generate the XML for image {}'.format(file_name))
            print(e)

from pylabel import importer
dataset = importer.ImportVOC(path='bike/annotations')
dataset.export.ExportToYoloV5()
from PIL import Image
import os, sys
import click

def resize(input_dir, output_dir):
	try:
		os.makedirs(output_dir)
	except OSError:
		pass

	dirs = os.listdir( input_dir )
	for item in dirs:
		if os.path.isfile(input_dir+item):
			im = Image.open(input_dir+item)
			f, e = os.path.splitext(input_dir+item)
			f = f.rpartition('/')[-1]
			imResize = im.resize((224,224), Image.ANTIALIAS)
			imResize.save(output_dir + f + '.tiff', 'TIFF')
			print(output_dir + f + '.tiff')

@click.command()
@click.option('--input_dir', default=None)
@click.option('--output_dir', default=None)
@click.option('--database', default='melanoma')
def main(input_dir, output_dir, database):
	if database == 'melanoma':
		resize(input_dir, output_dir)
	#else:

if __name__ == '__main__':
    main()

import sys
import cairosvg
from subprocess import call

source = sys.argv[1]
element = sys.argv[2]
target = sys.argv[3]

command = "../bin/phantomjs ../lib/extract.js \"" + source + "\" " + element + " >> \"" + target + ".svg\""

call(command, shell=True)

svg = open(target + ".svg").read()
fout = open(target + ".png",'w')
cairosvg.svg2png(bytestring=svg,write_to=fout)

svg = open(target + ".svg").read()
fout = open(target + ".pdf",'w')
cairosvg.svg2pdf(bytestring=svg,write_to=fout)

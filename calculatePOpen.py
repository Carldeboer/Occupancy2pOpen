#!/broad/software/free/Linux/redhat_5_x86_64/pkgs/python_2.7.1-sqlite3-rtrees/bin/python2.7
#doesn't work with:
#!/home/unix/cgdeboer/bin/python3
import itertools
import warnings
import subprocess
import MYUTILS
import numpy as np
import scipy as sp
from scipy.ndimage.filters import gaussian_filter;
from scipy import linalg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
import sys
import argparse
from bx.intervals.io import GenomicIntervalReader
from bx.bbi.bigwig_file import BigWigFile
parser = argparse.ArgumentParser(description='This program takes occupancy/openness BW tracks as input and outputs log(P(open)).')
parser.add_argument('-i',dest='inFP',	metavar='<inFile>',help='Input file containing a list of files, one per line with the following format, tab separated: \n<ID>\t<filePath>\t<gaussianSmoothingSD>\t<isOpen>\t<defaultValue>\t<doLog?>\n  where <isOpen> is 1 for openness tracks (e.g. FAIRE, DHS, ATAC-seq) and -1 for occupancy (e.g. nucleosomes)\n  and <doLog> is the number you want added to the data before log transform, or a negative number  otherwise', required=True);
parser.add_argument('-c',dest='chrsFile', metavar='<chrom.sizes>',help='A chrom.sizes file contiaining the sizes for all the chromosomes you want output', required=True);
parser.add_argument('-o',dest='outFPre', metavar='<outFilePre>',help='Where to output results, prefix [default=stdout]\n  Multiple files will be created, including a BW track, and diagnostics showing the clustering of data', required=False);
parser.add_argument('-l',dest='logFP', metavar='<logFile>',help='Where to output errors/warnings [default=stderr]', required=False);
parser.add_argument('-s',dest='sample', metavar='<sampleFrac>',help='The fraction of data points to use [default=0.01]', required=False, default = 0.01);
parser.add_argument('-d',dest='dim', metavar='<graphDim>',help='Inches per graph [default=3]', required=False, default = 3);
parser.add_argument('-v',dest='verbose', action='count',help='Verbose output?', required=False, default=0);
args = parser.parse_args();

# this is for dubugging in interactive mode - comment out normally
#args = lambda: None
#setattr(args,"sample","0.01")
#setattr(args,"dim","3")
#setattr(args,"inFP","calcPOpenTestData.txt")
#setattr(args,"outFPre","calcPOpenTestData_out")
#setattr(args,"chrsFile","/home/unix/cgdeboer/genomes/sc/20110203_R64/chrom.sizes");
#setattr(args,"logFP",None);
	

args.sample = float(args.sample);
args.dim = float(args.dim);

if (args.logFP is not None):
	logFile=MYUTILS.smartGZOpen(args.logFP,'w');
	sys.stderr=logFile;


IDs =[];
files = [];
smoothings = [];
isOpenness = [];
defaultVal = [];
doLog = [];
clusterInit = [];
inFile=MYUTILS.smartGZOpen(args.inFP,'r');
for line in inFile:
	if line is None or line == "" or line[0]=="#": continue
	data=line.rstrip().split("\t");
	IDs.append(data[0]);
	files.append(data[1]);
	smoothings.append(float(data[2]));
	isOpenness.append(int(data[3]));
	defaultVal.append(float(data[4]));
	doLog.append(float(data[5]));
	cI = float(data[6]);
	if float(data[5])>=0:
		cI=np.log10(cI+ float(data[5]));
	clusterInit.append(cI); 

inFile.close();

#get loci of interest and their sizes
chromSizesFile = MYUTILS.smartGZOpen(args.chrsFile,'r');
chromSizes = {};
chrOrder = [];
for line in chromSizesFile:
	if line is None or line == "" or line[0]=="#": continue
	data=line.rstrip().split("\t");
	chromSizes[data[0]]=int(data[1]);
	chrOrder.append(data[0]);
	

#determine what positions will be used in the data
useThese = {};
totalLength = 0
for chr in chrOrder:
	#sample positions
	useThese[chr] = np.random.random_sample((chromSizes[chr]))<args.sample;
	totalLength = totalLength + np.sum(useThese[chr]);


#make a matrix of the data
allData = np.empty([totalLength,len(IDs)]);
for i in range(0,len(IDs)):
	#input GB tracks
	curBW = BigWigFile(open(files[i]))
	curTot = 0;
	for chr in chrOrder:
		values = curBW.get_as_array( chr, 0, chromSizes[chr] )
		#fill in blanks
		invalid = np.isnan( values )
		values[ invalid ] = defaultVal[i];
		#log transform
		if doLog[i]>=0:
			values =np.log10(values + doLog[i])
		#smooth data
		if smoothings[i]!=0:
			#reflect = at edge of array, the data will be mirrored
			#truncate = don't incorporate data more than X away - probably a great speed increase since the distrib goes out to infinity
			values = gaussian_filter(values, smoothings[i], mode='reflect', truncate=4.0)
		#sample data
		values = values [useThese[chr]]
		#append data;
		allData[curTot:(curTot+len(values)),i] = values;
		curTot = curTot + len(values);

dataMedians = np.median(allData,axis=0);
initialMeans = np.asarray([dataMedians,clusterInit]);
#learn MOG
#n_iter = num iterations for EM
#n_init = num re-initiations
#params = 'w'eights, 'm'eans, 'c'ovariance
#covariance_type: full = full covariance, diag = only variance modeled, also other options
myGMM = mixture.GMM(n_components=2, covariance_type='full', n_iter=100, n_init=100, params='wmc', init_params='wc')
myGMM.means_=initialMeans;
myGMM.fit(allData);
myDPGMM = mixture.DPGMM(n_components=2, covariance_type='full', n_iter=100, params='wmc', init_params='wc')
myDPGMM.means_=initialMeans;
myDPGMM.fit(allData);
# .weights_, .means_, .covars_

#print a graph of the clustering
for j, (myMM, title) in enumerate([(myGMM,"GMM"),(myDPGMM,"DPGMM")]):
	plotOrder = np.argsort(-myMM.weights_); # from highest to lowest weight (so that overlapping data will still be visible
	plt.figure(figsize=(args.dim*len(IDs),args.dim*len(IDs)),dpi=300);
	cluster = myMM.predict(allData)
	for x in range(0,len(IDs)):
		for y in range(0,len(IDs)): # each pair of inputs
			splot = plt.subplot(len(IDs),len(IDs),x*len(IDs)+y+1)
			color_iter = itertools.cycle(['g', 'r', 'c', 'm'])
			for k in plotOrder: # scatter for each cluster
				colour = color_iter.next();
				plt.scatter(allData[cluster==k,y],allData[cluster==k,x],color = colour, alpha=0.1)
				#if 1:
			for k in plotOrder: # ellipse for each cluster
				colour = color_iter.next();
				if title=="GMM":
					covs =myMM.covars_[k,[[y,y],[x,x]],[[y,x],[y,x]]];
				else:
					covs =myDPGMM._get_covars();
					covs =covs[k][[[y,y],[x,x]],[[y,x],[y,x]]];
				v,w = linalg.eigh(covs); # split covariance into eigenvectors (w) and values (v)
				u = w[0]/linalg.norm(w[0]); #the Frobenius norm of the eigenvector
				#plt.scatter(allData[cluster==k,y],allData[cluster==k,x],s=0.8, color = 'k', alpha=0.5)
				#plot elipses
				#angle = np.arctan(u[x]/u[y]);
				angle = np.arctan2(w[0][1], w[0][0]);
				angle = 180 * angle/np.pi #degrees
				#v*=4;
				ell = matplotlib.patches.Ellipse(myMM.means_[k,[y,x]], v[0], v[1],180 + angle, color = colour);
				ell.set_clip_box(splot.bbox);
				ell.set_alpha(0.5);
				splot.add_artist(ell);
			if y==0:
				plt.ylabel(IDs[x]);
			if x==(len(IDs)-1):
				plt.xlabel(IDs[y]);
	plt.savefig('%s_%s_scatters.png'%(args.outFPre,title))
#plt.savefig('%s_GMM_scatters.pdf'%args.outFPre)




#pick out what cluster is the one I want
#'open' should have high means when isOpenness is 1 and low otherwise
#should also be the smallest cluster
allMeans = myGMM.means_;

meanDiffs = allMeans - dataMedians;
directions = np.array([isOpenness,]*meanDiffs.shape[0]) * meanDiffs;
openClust = -1;
for i in range(0,directions.shape[0]):
	if np.all(directions[i]>0):
		if myGMM.weights_[i]==np.min(myGMM.weights_):
			if openClust<0:
				sys.stderr.write("Multiple candidate openClusters: %i and %i\n"%(i,openClust));
			openClust=i;
		else:
			sys.stderr.write("All in right direction, but not smaller cluster: %i\n"%(i));
			if openClust<0:
				openClust=i

if openClust<0:
	openClust = np.argsort(myGMM.weights_)[0];				
	sys.stderr.write("None have all in right direction, taking smallest cluster: %i\n"%(openClust));

#one should be negative and one positive
sys.stderr.write("Using cluster %i as 'open', with %f%% of the data\n"%(openClust,myGMM.weights_[openClust]*100));

#re-input all data and calculate P(open)



allBWs = [];
for i in range(0,len(IDs)):
	#input GB tracks
	curBW = BigWigFile(open(files[i]))
	allBWs.append(curBW);


toBW = subprocess.Popen(["wigToBigWig","stdin",args.chrsFile,"%s_pOpen.bw"%(args.outFPre)], stdin=subprocess.PIPE)
toBW.stdin.write("track type=wiggle_0\n")


for chr in chrOrder:
	chrData  = np.empty([chromSizes[chr],len(IDs)]);
	for i in range(0,len(IDs)):
		values = allBWs[i].get_as_array( chr, 0, chromSizes[chr] )
		#fill in blanks
		invalid = np.isnan( values )
		values[ invalid ] = dataMedians[i]; # at this stage, replace missing values with median
		#log transform
		if doLog[i]>=0:
			values =np.log10(values + doLog[i])
		#smooth data
		if smoothings[i]!=0:
			#reflect = at edge of array, the data will be mirrored
			#truncate = don't incorporate data more than X away - probably a great speed increase since the distrib goes out to infinity
			values = gaussian_filter(values, smoothings[i], mode='reflect', truncate=4.0)
		#add data;
		chrData[:,i] = values;
	pred = myGMM.predict_proba(chrData);
	toBW.stdin.write("fixedStep chrom=%s start=1 step=1\n"%(chr))
	toBW.stdin.write("\n".join(map(str,pred[:,openClust])));
	toBW.stdin.write("\n");

temp = toBW.communicate();
if temp[0] is not None:
	sys.stderr.write(temp[0]);

if temp[1] is not None:
	sys.stderr.write(temp[1]);

#raise Exception("Reached bad state=%d for '%s.%d' '%s' at line '%s'" %(state,mid,ver,tfid,line));
if (args.logFP is not None):
	logFile.close();

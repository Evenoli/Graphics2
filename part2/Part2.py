import sys
import numpy as np
from PNM import *

def CreateAndSavePFM(out_path):
    width = 512
    height = 512
    numComponents = 3

    img_out = np.empty(shape=(width, height, numComponents), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = 1.0

    writePFM(out_path, img)

def LoadAndSavePPM(in_path, out_path):
    img_in = loadPPM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    height,width,_ = img_in.shape # Retrieve height and width
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] # Copy pixels

    writePPM(out_path, img_out)

def LoadAndSavePFM(in_path, out_path):
    img_in = loadPFM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=img_in.dtype)
    height,width,_ = img_in.shape # Retrieve height and width
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:] # Copy pixels

    writePFM(out_path, img_out)

def LoadPPMAndSavePFM(in_path, out_path):
    img_in = loadPPM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            img_out[y,x,:] = img_in[y,x,:]/255.0

    writePFM(out_path, img_out)
            
def LoadPFMAndSavePPM(in_path, out_path):
    img_in = loadPFM(in_path)
    img_out = np.empty(shape=img_in.shape, dtype=np.float32)
    height,width,_ = img_in.shape
    for y in range(height):
        for x in range(width):
            [a, b, c] = img_in[y,x,:]
            if a > 1:
                a = 1.0
            a = a * 255.0
            if b > 1:
                b = 1.0
            b = b * 255.0
            if c > 1.0:
                c = 1.0
            c = c * 255.0
            img_out[y, x] = [a, b, c]

    writePPM(out_path, img_out.astype(np.uint8))

def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0: 
       return v
    return v/norm

#Part 2
def GenerateSamples(numberOfSamples):
    #Load Grace Environment map
    em = loadPFM("grace_latlong.pfm")
    height, width,_ = em.shape
    em_intensity = np.empty(shape=(height, width, 1))
    #fill em_intensity with em values st I = R+G+B /3
    # colTotals = np.empty(shape=(width, 1))
    rowTotals = np.zeros(shape=(height, 1))
    print "Bringing map into intensity space"
    for row in range(height):
        for col in range(width):
            em_intensity[row, col] = float((em[row, col, 0] + em[row, col, 1] + em[row, col, 2])) /3.0
            #Scale by solid angle term sin(theta)
            theta = (float(row)/float(height)) * np.pi
            em_intensity[row, col] = em_intensity[row, col] * np.sin(theta)
            # print em_intensity[row, col]
            # colTotals[col] = colTotals[col] + em_intensity[row, col]
            rowTotals[row] = rowTotals[row] + em_intensity[row, col]
    #Create 1D CDF across rows to select one specific row
    print "Creating across row pdf/cdf"
    acrossRowSum = 0.0
    for i in range(height):
        acrossRowSum = acrossRowSum + rowTotals[i]
    #Generate PDF --------------------------------------------------------->NEEDED IN PART 3 as p(x)
    chooseRowPDF = np.zeros(shape=(height, 1))
    for i in range(height):
        chooseRowPDF[i] = rowTotals[i]/acrossRowSum
    #Generate CDF
    chooseRowCDF = np.zeros(shape=(height, 1))
    for i in range(height):
        if i != 0:
            chooseRowCDF[i] = chooseRowPDF[i] + chooseRowCDF[i-1]
        else:
            chooseRowCDF[i] = chooseRowPDF[i]


    #Seperately create one 1D CDF for every row of the EM to select a pixel within that row
    print "Creating per row cdf/pdf"
    per_row_pdf = np.empty(shape=(height, width, 1))
    for i in range(height):
        for j in range(width):
            #rowTotals[0] = 0 as sin(0) = 0 so not dividing
            if i == 0:
                per_row_pdf[i, j] = 1.0/width
            else:
                per_row_pdf[i, j] = em_intensity[i, j]/rowTotals[i]
    #Generate CDF
    per_row = np.empty(shape=(height, width, 1))
    for col in range(width):
        for row in range(height):
            if col != 0:
                per_row[row, col] = per_row_pdf[row, col] + per_row[row, col-1]
            else:
                per_row[row, col] = per_row_pdf[row, col]

    neighbours = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                               for y2 in range(y-1, y+2)
                               if (-1 < x < height and
                                   -1 < y < width and
                                   (x != x2 or y != y2) and
                                   (0 <= x2 < height) and
                                   (0 <= y2 < width))]

    #Begin sampling
    print "Sampling" + str(numberOfSamples)
    randRow = np.random.rand(numberOfSamples)
    sampledIndices = np.zeros(shape=(numberOfSamples, 2))
    for s in range(numberOfSamples):
        i = 0
        found = False
        while not found:
            if chooseRowCDF[i] >= randRow[s]:
                sampledIndices[s, 1] = int(i)
                found = True
            i = i+1
        i = 0
        found = False
        randPixelInRow = np.random.rand(1)
        while not found:
            if per_row[sampledIndices[s, 1], i] >= randPixelInRow[0]:
                sampledIndices[s, 0] = int(i)
                found = True
            i = i+1
        #Highlight pixel in pfm map
        row = sampledIndices[s, 1]
        col = sampledIndices[s, 0]
        em[row, col] = [0,0,1.0]
        #highlight neightbours
        pixels = neighbours(int(row), int(col))
        for i in range(len(pixels)):
            em[pixels[i]] = [0, 0, 1.0]
            nextPixels = neighbours(pixels[i][0], pixels[i][1])
            for j in range(len(nextPixels)):
                em[nextPixels[j]] = [0,0,1.0]

    writePFM("sampled_map_" + str(numberOfSamples) + ".pfm", em)
    LoadPFMAndSavePPM("sampled_map_" + str(numberOfSamples) + ".pfm", "sampled_map_" + str(numberOfSamples) + ".ppm")

    return sampledIndices, chooseRowCDF, per_row_pdf

#Part 3
def RenderDiffuseSphere(numberOfSamples, sampledIndices, chooseRowCDF, per_row_pdf):
    img_in = loadPFM("grace_latlong.pfm")
    print("Building diffuse shpere")
    radius = 255.5
    sphereWidth = 511
    sphereHeight = 511
    sphere = np.empty((sphereWidth, sphereHeight, 3), dtype=np.float32)

    height, width, _ = img_in.shape

    for w in range(sphereWidth):
        for h in range(sphereHeight):
            if (h - radius)**2 + (w - radius)**2 <= radius**2:
                x = (w - radius)/radius
                y = (radius - h)/radius
                z = np.sqrt(1.0 - x**2 - y**2)
                normal = normalize([x, y, z])

                pixelSum = [0,0,0]
                for s in range(numberOfSamples):
                    i, j = sampledIndices[s]
                    #Polar (t) and azimuthal (p) angles
                    t = (float(j)/float(height))*np.pi
                    p = (float(i)/float(width))*np.pi*2

                    #Reflection vector
                    [a,b,c] = normalize([-np.sin(t)*np.sin(p)/radius, np.cos(t)/radius, -np.cos(p)*np.sin(t)/radius])

                    cosTheta = np.dot(normal, [a,b,c])

                    pxy = chooseRowCDF[j] * per_row_pdf[j,i]
                    
                    if(cosTheta > 0):
                        pixelSum[:] += normalize(((1.0/np.pi) * cosTheta * img_in[j,i,:]) / pxy)

                sphere[w,h,0] = pixelSum[0] / float(numberOfSamples)
                sphere[w,h,1] = pixelSum[1] / float(numberOfSamples)
                sphere[w,h,2] = pixelSum[2] / float(numberOfSamples)
                # print(sphere[w,h])

            else:
                sphere[h,w,:] = np.array([0.0, 0.0, 0.0])

    writePFM("part3_"+str(numberOfSamples)+".pfm", sphere)
    LoadPFMAndSavePPM("part3_"+str(numberOfSamples)+".pfm", "part3_"+str(numberOfSamples)+".ppm")

indicies,rowPDF,perRowPDF = GenerateSamples(64)
RenderDiffuseSphere(64, indicies,rowPDF,perRowPDF)

indicies,rowPDF,perRowPDF = GenerateSamples(256)
RenderDiffuseSphere(256, indicies,rowPDF,perRowPDF)

indicies,rowPDF,perRowPDF = GenerateSamples(1024)
RenderDiffuseSphere(1024, indicies,rowPDF,perRowPDF)
#GenerateSamples(256)
#GenerateSamples(1024)
if '__main__' == __name__:
    pass

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

def magnitude(v):
    return math.sqrt(sum(v[i]*v[i] for i in range(len(v))))

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

    return sampledIndices, chooseRowPDF, per_row_pdf

#Part 3
def RenderDiffuseSphere(numberOfSamples, sampledIndices, chooseRowPDF, per_row_pdf, useProb):
    img_in = loadPFM("grace_latlong.pfm")
    print("Building diffuse shpere")
    radius = 255.5
    sphereWidth = 511
    sphereHeight = 511
    sphere = np.zeros((sphereWidth, sphereHeight, 3), dtype=np.float32)

    height, width, _ = img_in.shape

    for w in range(sphereWidth):
        if(w%50 == 0):
            print(w)
        for h in range(sphereHeight):
            if (h - radius)**2 + (w - radius)**2 <= radius**2:
                x = (w - radius)/radius
                y = (radius - h)/radius
                z = np.sqrt(1.0 - x**2 - y**2)
                normal = normalize(np.array([x, y, z]))

                pixelSum = np.zeros(3, dtype=np.float32)

                for s in range(numberOfSamples):
                    i, j = sampledIndices[s]
                    #Polar (theta) and azimuthal (phi) angles
                    theta = (float(j)/float(height))*np.pi
                    phi = (float(i)/float(width))*np.pi*2

                    a = -np.sin(theta)*np.cos(phi)*radius
                    b = np.cos(theta)*radius
                    c = np.sin(theta)*np.sin(phi)*radius

                    cosTheta = np.dot(normal, normalize(np.array([a,b,c])))

                    if(useProb):
                        if(cosTheta > 0):
                            pxy = chooseRowPDF[j] * per_row_pdf[j,i]
                            
                            I = (img_in[j,i,0] + img_in[j,i,1] + img_in[j,i,2]) / float(3)
                            norm = np.sqrt(img_in[j,i,0]**2 + img_in[j,i,1]**2 + img_in[j,i,2]**2)
                            colNorm = np.zeros(3)
                            colNorm[0] = img_in[j,i,0] / norm
                            colNorm[1] = img_in[j,i,1] / norm
                            colNorm[2] = img_in[j,i,2] / norm

                            v = (1.0/np.pi) * cosTheta * I * colNorm[:] / pxy
                            pixelSum += v #/ np.sum()
                    else:
                        if(cosTheta > 0):
                            v = (1.0/np.pi) * cosTheta * img_in[j,i]
                            pixelSum += v #/ np.sum(v)

                sphere[w,h] = pixelSum / float(numberOfSamples)

            else:
                sphere[h,w,:] = np.array([0.0, 0.0, 0.0])

    if(useProb):
        writePFM("spherePxy"+str(numberOfSamples)+".pfm", sphere)
        LoadPFMAndSavePPM("spherePxy"+str(numberOfSamples)+".pfm", "spherePxy"+str(numberOfSamples)+".ppm")
    else:
        writePFM("sphere"+str(numberOfSamples)+".pfm", sphere)
        LoadPFMAndSavePPM("sphere"+str(numberOfSamples)+".pfm", "sphere"+str(numberOfSamples)+".ppm")

def LinearToneMap(img_in, img_out):
    F = loadPFM(img_in)
    n = np.empty(shape=F.shape, dtype=np.float32)

    print("Performing tone mapping")
    dimmest = 1.0
    brightest = 0.0

    height, width, _ = F.shape
    for y in range(width):
        for x in range(height):
            intensity = np.sum(F[x][y])/3.0
            if (intensity < dimmest and intensity != 0.0):
                dimmest = intensity
            if (intensity > brightest):
                brightest = intensity

    print("Brightest pixel: ", brightest, " Dimmest pixel: ", dimmest )

    for y in range(width):
        for x in range(height):
            for i in range(0,3):
                n[x,y,i] = F[x,y,i] / brightest

    writePFM(img_out + ".pfm", n)
    LoadPFMAndSavePPM(img_out+".pfm",  img_out + ".ppm")

def Exposure(img_in, img_out, stops):
    print("Performing exposure mapping")
    scale = np.power(2.0, stops)
    F = loadPFM(img_in)
    n = np.empty(shape=F.shape, dtype=np.float32)

    height, width, _ = F.shape
    for y in range(width):
        for x in range(height):
            for i in range(0,3):
                intensity = F[x,y,i] * scale
                if(intensity > 1.0):
                    intensity = 1.0
                n[x,y,i] = intensity

    writePFM(img_out + ".pfm", n)
    LoadPFMAndSavePPM(img_out+".pfm",  img_out + ".ppm")

#=========================================================================================================64

#numSamples = 64
#Generate Samples using Environment map
#indicies,rowPDF,perRowPDF = GenerateSamples(numSamples)
#Generate Sphere without using probabilities
#RenderDiffuseSphere(numSamples, indicies,rowPDF,perRowPDF, False)
#Exposure("sphere"+str(numSamples)+".pfm", "sphere"+str(numSamples)+"expo", -2)
#LinearToneMap("sphere"+str(numSamples)+".pfm", "sphere"+str(numSamples)+"expotone")

#Generate Sphere using probabilities
#RenderDiffuseSphere(numSamples, indicies,rowPDF,perRowPDF, True)
#Exposure("spherePxy"+str(numSamples)+".pfm", "spherePxy"+str(numSamples)+"expo", -15)
#LinearToneMap("spherePxy"+str(numSamples)+".pfm", "spherePxy"+str(numSamples)+"expotone")

#=========================================================================================================256

#numSamples = 256
#Generate Samples using Environment map
#indicies,rowPDF,perRowPDF = GenerateSamples(numSamples)
#Generate Sphere without using probabilities
#RenderDiffuseSphere(numSamples, indicies,rowPDF,perRowPDF, False)
#Exposure("sphere"+str(numSamples)+".pfm", "sphere"+str(numSamples)+"expo", -2)
#LinearToneMap("sphere"+str(numSamples)+".pfm", "sphere"+str(numSamples)+"expotone")

#Generate Sphere using probabilities
#RenderDiffuseSphere(numSamples, indicies,rowPDF,perRowPDF, True)
#Exposure("spherePxy"+str(numSamples)+".pfm", "spherePxy"+str(numSamples)+"expo", -2)
#LinearToneMap("spherePxy"+str(numSamples)+".pfm", "spherePxy"+str(numSamples)+"expotone")

#=========================================================================================================1024

numSamples = 1024
#Generate Samples using Environment map
#indicies,rowPDF,perRowPDF = GenerateSamples(numSamples)
#Generate Sphere without using probabilities
#RenderDiffuseSphere(numSamples, indicies,rowPDF,perRowPDF, False)
#Exposure("sphere"+str(numSamples)+".pfm", "sphere"+str(numSamples)+"expo", -2)
#LinearToneMap("sphere"+str(numSamples)+".pfm", "sphere"+str(numSamples)+"expotone")

#Generate Sphere using probabilities
#RenderDiffuseSphere(numSamples, indicies,rowPDF,perRowPDF, True)
Exposure("spherePxy"+str(numSamples)+".pfm", "spherePxy"+str(numSamples)+"expo", -15)
LinearToneMap("spherePxy"+str(numSamples)+".pfm", "spherePxy"+str(numSamples)+"toneexpotone")


if '__main__' == __name__:
    pass

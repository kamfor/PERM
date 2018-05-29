#!/usr/bin/env python
# -*- coding: utf8 -*-

import freenect
import cv2
import frame_convert2
import datetime
import numpy as np



def get_depth():
	return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
	return frame_convert2.video_cv(freenect.sync_get_video()[0])

def nothing(x):
	pass
    

def setupMask():
	cv2.namedWindow('(2) Mask')
	cv2.createTrackbar('Size','(2) Mask',1,5,nothing)
	cv2.createTrackbar('Thr','(2) Mask',128,255,nothing)
	
def setupContours():
	cv2.namedWindow('(3) Contours')
	cv2.createTrackbar('MinSize','(3) Contours',1000,100000,nothing)

def preprocess(img):
	dst = cv2.GaussianBlur(img,(5,5),0)
	cv2.imshow('(1) Preprocess', dst)
	return dst
	
def createMask(img):
	# zmiana na odcienie szarości
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	# progowanie obrazu 
	thr = cv2.getTrackbarPos('Thr','(2) Mask')
	th,mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)

	# otwarcie morfologiczne - pozbycie się drobnego szumu
	kernel_size = cv2.getTrackbarPos('Size','(2) Mask')
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*kernel_size+1,2*kernel_size+1))
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	
	cv2.imshow('(2) Mask', mask)
	return mask

def detectContours(img, mask):
	img_out = img.copy()

	# detekcja konturów na podstawie obrazu maski
	im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# narysowanie wszystkich konturów (czerwone)
	cv2.drawContours(img_out, contours, -1, (0,0,255), 1)


	# pobranie wartości suwaka 
	min_size = cv2.getTrackbarPos('MinSize','(3) Contours')

	# filtrowanie konturów i wybranie największego
	maxarea = -1
	filtered_cnt = []
	maxcnt = []
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if area > min_size:
			filtered_cnt.append(cnt)

		if area > maxarea and area > min_size:
			maxarea = area 
			maxcnt = cnt

	# narysowanie odfiltrowanych konturów (niebieski)
	cv2.drawContours(img_out, filtered_cnt, -1, (255, 0,0), 2)

	# analiza największego konturu
	if maxarea > 0:
		cv2.drawContours(img_out, [maxcnt], 0, (0,255,0), 3)
		
	cv2.imshow('(3) Contours', img_out)
	return maxcnt

def calculateFeatures(img, cnt):
	# w przypadku braku konturu
	if len(cnt) == 0:
		return [99999999]
		
	# podstawowe cechy konturu
	area = cv2.contourArea(cnt)
	perimeter = cv2.arcLength(cnt, True)
	(x, y), (w, h), angle = cv2.minAreaRect(cnt)

	# wyznaczenie momentów geometrycznych
	M = cv2.moments(cnt)
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	
	Hm = cv2.HuMoments(M).flatten()

	# przygotowanie wektora cech
	#f1 = area/(w*h)
	f1 = Hm[0]
	f2 = w/h
	f3 = area/(w*h)
	ret = [f1,f2,f3]

	return ret


# trening modelu klasyfikatora
def train(features, labels):
	# dla klasyfikatora NN nie wymaga nic ponad podane dane
	# liczymy srodek ciezkosci dla modelu 
	print features
	print model["features"]
	s1 = 0
	s2 = 0
	s3 = 0

	features  = [s1/i,s2/i,s3/i]
	
	
	return {"features": features, "labels": labels}

# rozpoznawanie z użyciem wygenerowanego modelu
def classify(model, feature):
	min_dist = 100000000.0
	min_id = -1
	# wyszkukiwanie najbliższej próbki w zbiorze uczącym
	for i in range(len(model["features"])):
		dist = np.linalg.norm(np.array(model["features"][i]) - np.array(feature))
		if dist < min_dist:
			min_dist = dist
			min_id = i

	# jeśli znaleziono - zwracana jest jej etykieta
	if min_id >= 0:
		return model["labels"][min_id], min_dist
	else:
		return 0, 0




def main():
	setupMask()
	setupContours()

	features = []
	labels = []
	
	

	label_names = ["UNKNOWN", "object_1", "object_2", "object_3"]

	do_classify = False

	print('Press ESC in window to stop')
	while 1:
		img = get_video()

		# przetwarzanie wstępne obrazu kolorowego
		img_prep = preprocess(img)
		# wyznaczenie maski
		mask = createMask(img_prep)

		# detekcja konturów
		cnt = detectContours(img, mask)

		# wyznaczenie cech obiektu
		f = calculateFeatures(img_prep, cnt)

		if do_classify:
			lbl, dist = classify(model, f)
			print(str(f) + " : " + label_names[lbl] + "(dist = " + str(dist) + ")")


		# wyświetlenie obrazu wejściowego
		cv2.imshow('(0) Video', img)
		
		# wait for key press
		ch = cv2.waitKey(10)

		# nothing pressed
		if ch < 0:
			continue

		# ESC - end process
		if ch == 27:
			break
		# s - save picture
		if chr(ch) == 's':
			now = datetime.datetime.now()
			fname = now.strftime("%Y%m%d_%H%M%S%f.png");
			cv2.imwrite(fname, img)
			print("Saved: " + fname)

		# 1 - save first object
		if chr(ch) == '1':
			features.append(f)
			labels.append(1)
			print("Zapamiętano: " + label_names[1])
		# 2 - save second object
		if chr(ch) == '2':
			features.append(f)
			labels.append(2)
			print("Zapamiętano: " + label_names[2])
		# 3 - save third object
		if chr(ch) == '3':
			features.append(f)
			labels.append(3)
			print("Zapamiętano: " + label_names[3])

		# przełącz tryb klasyfikacji
		if chr(ch) == 't':
			model = train(features, labels)
			do_classify = True


	cv2.destroyAllWindows()
        
if __name__ == "__main__":
	main()

import os
import cv2
import face_recognition as fg

os.chdir("C:/Users/abdul/OneDrive/Pictures/New folder")

known_img = cv2.imread("usama.jpg")


# Load a sample picture and learn how to recognize it.
usama_image = fg.load_image_file("usama.jpg")
usama_face_encoding = fg.face_encodings(usama_image)[0]

# Create arrays of known face encodings
known_face_encodings = [usama_face_encoding]


# Load an image with an unknown face

unknown_image = fg.load_image_file("test.jpg")
unknown_face_encoding = fg.face_encodings(unknown_image)[0]

# Calculate face distance
face_distances = fg.face_distance(known_face_encodings, unknown_face_encoding)

print("The face has a distance of {:.2} from the face in unknown ".format(face_distances[0]))
print("With cutoff of less than 0.6, Match is {}".format(face_distances[0] < 0.6))

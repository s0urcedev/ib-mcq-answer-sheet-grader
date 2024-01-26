import cv2
import numpy as np
import fitz
import img2pdf
import PIL
import io

template = cv2.imread('template.jpg')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(500)
template_kps, template_descs = orb.detectAndCompute(template_gray, None)

def process_image_to_bytes_io(image, bytes_io, correct_answers):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_kps, image_descs = orb.detectAndCompute(image_gray, None)

    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = sorted(matcher.match(image_descs, template_descs, None), key=lambda x: x.distance)
    matches = matches[:int(len(matches) * 0.2)]

    image_pts = np.zeros((len(matches), 2), dtype="float")
    template_pts = np.zeros((len(matches), 2), dtype="float")

    for i, m in enumerate(matches):
        image_pts[i] = image_kps[m.queryIdx].pt
        template_pts[i] = template_kps[m.trainIdx].pt

    H, mask = cv2.findHomography(image_pts, template_pts, method=cv2.RANSAC)

    h, w = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    level_coordiantes = ((685, 635), (910, 635), (990, 660))
    level_dimensions = (20, 20)
    roi = (thresh[level_coordiantes[0][1]:level_coordiantes[0][1] + level_dimensions[1], level_coordiantes[0][0]:level_coordiantes[0][0] + level_dimensions[0]], thresh[level_coordiantes[1][1]:level_coordiantes[1][1] + level_dimensions[1], level_coordiantes[1][0]:level_coordiantes[1][0] + level_dimensions[0]])
    non_zeros = (cv2.countNonZero(roi[0]), cv2.countNonZero(roi[1]))
    if non_zeros[0] > non_zeros[1]:
        level = 0
    else:
        level = 1
    levels = ['HL', 'SL']
    cv2.putText(aligned, levels[level], level_coordiantes[2], cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 0, 0), 7, cv2.LINE_AA, False)

    box_coordinates = (250, 1048)
    box_dimensions = (30, 20)
    box_steps = (75, 70, 412)

    answers = []
    total_correct = 0

    for c in range(0, 3):
        for y in range(0, 14):
            if c == 2 and y == 12:
                break
            mx = -1
            mxi = -1
            for x in range(0, 4):
                current_coordinates = (box_coordinates[0] + box_steps[0] * x + box_steps[2] * c, box_coordinates[1] + box_steps[1] * y)
                roi = thresh[current_coordinates[1]:current_coordinates[1]+box_dimensions[1], current_coordinates[0]:current_coordinates[0]+box_dimensions[0]]
                non_zeros = cv2.countNonZero(roi)
                if mx < non_zeros:
                    mx = non_zeros
                    mxi = x
            if mx > 10:
                answers.append(chr(ord('A') + mxi))
            else:
                answers.append(None)
            print(f"{len(answers)}. {answers[-1]}")
            if len(answers) - 1 < len(correct_answers[level]):
                line_coordinates = (box_coordinates[0] + box_steps[0] * 4 + box_steps[2] * c, box_coordinates[1] + box_steps[1] * y)
                text_coordinates = (box_coordinates[0] + box_steps[0] * mxi + box_steps[2] * c, box_coordinates[1] + box_steps[1] * y + box_dimensions[1] + box_dimensions[1] // 2)
                correct_text_coordinates = (box_coordinates[0] + box_steps[0] * (ord(correct_answers[level][len(answers) - 1]) - ord('A')) + box_steps[2] * c, box_coordinates[1] + box_steps[1] * y + box_dimensions[1] + box_dimensions[1] // 2)
                if answers[-1] is not None:
                    if answers[-1] == correct_answers[level][len(answers) - 1]:
                        total_correct += 1
                        cv2.line(aligned, (line_coordinates[0], line_coordinates[1]), (line_coordinates[0] + box_dimensions[0] // 2, line_coordinates[1] + box_dimensions[1]), (0, 200, 0), 9)
                        cv2.line(aligned, (line_coordinates[0] + box_dimensions[0] // 2, line_coordinates[1] + box_dimensions[1]), (line_coordinates[0] + box_dimensions[0], line_coordinates[1]), (0, 200, 0), 9)
                        cv2.putText(aligned, chr(ord('A') + mxi), text_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 7, cv2.LINE_AA, False)
                    else:
                        cv2.line(aligned, (line_coordinates[0], line_coordinates[1]), (line_coordinates[0] + box_dimensions[0], line_coordinates[1] + box_dimensions[1]), (0, 0, 200), 9)
                        cv2.line(aligned, (line_coordinates[0] + box_dimensions[0], line_coordinates[1]), (line_coordinates[0], line_coordinates[1] + box_dimensions[1]), (0, 0, 200), 9)
                        cv2.putText(aligned, chr(ord('A') + mxi), text_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 7, cv2.LINE_AA, False)
                        cv2.putText(aligned, correct_answers[level][len(answers) - 1], correct_text_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 7, cv2.LINE_AA, False)
                else:
                    cv2.line(aligned, (line_coordinates[0], line_coordinates[1]), (line_coordinates[0] + box_dimensions[0], line_coordinates[1] + box_dimensions[1]), (0, 0, 200), 9)
                    cv2.line(aligned, (line_coordinates[0] + box_dimensions[0], line_coordinates[1]), (line_coordinates[0], line_coordinates[1] + box_dimensions[1]), (0, 0, 200), 9)
                    cv2.putText(aligned, correct_answers[level][len(answers) - 1], correct_text_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 7, cv2.LINE_AA, False)

    total_text_coordinates = (box_coordinates[0] + box_steps[0] + box_steps[1] // 2 + box_steps[2] * 2, box_coordinates[1] + box_steps[1] * 13)
    cv2.putText(aligned, f"{total_correct}/{len(correct_answers[level])}", total_text_coordinates, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (200, 0, 0), 7, cv2.LINE_AA, False)

    pil_image = PIL.Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    pil_image.save(bytes_io, format='JPEG')

def grade(bytes, correct_answers):
    outputs = []
    for idx, page in enumerate(fitz.open(stream=bytes, filetype='pdf')): 
        pix = page.get_pixmap()
        bytes = np.frombuffer(pix.samples, dtype=np.uint8)
        image = bytes.reshape(pix.height, pix.width, pix.n)
        bytes_io = io.BytesIO()
        process_image_to_bytes_io(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), bytes_io, correct_answers)
        outputs.append(bytes_io.getvalue())
    return img2pdf.convert(outputs)
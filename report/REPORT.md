# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 2026-11-06

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:** Python 3.13.18 / PyTorch 2.11.0+cpu / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt` (sha256 `c62d41bf9625…`), `yolo11n.pt` (sha256 `0ebbc80d4a76…`), `yolo11n-seg.pt` (sha256 `55ed65c56c91…`)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không. Chạy nguyên bản `v1.0.3`; không đổi checkpoint, không đổi threshold, không sửa mã nguồn.

**Sample dùng trong báo cáo:** `traffic` = COCO 210273 (640 x 428) cho mục 1; `kitchen` = COCO 397133 cho mục 2 và 3. Ô validation cuối notebook báo `PASS` cho 3 JSON, 3 PNG, attribution và `instance_id` duy nhất.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số "tin" của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

**1.** Record hạng 1 của `sample_id = traffic`:
`class_id` = 468 ; `class_name` = cab ; `rank` = 1 ; `score` = 0.510915 ; `taxonomy_name` = ImageNet-1K.

**2.** Record này nói về cả bức ảnh, không nói về một vật thể riêng. Trong record không có `bbox_xyxy`
cũng không có `polygon_xy`, nên không có chỗ nào chỉ ra vật thể đó nằm ở đâu. Hai trường toạ độ duy nhất
là `image_width` = 640 và `image_height` = 428, tức là kích thước toàn ảnh. Ảnh `traffic` chỉ sinh ra 5
record ứng với `rank` 1 đến 5, và cả 5 đều là tên một loại xe: cab (0.510915), minibus (0.164284),
police_van (0.085848), recreational_vehicle (0.054110), streetcar (0.048193). Đó là 5 cách gọi cho cùng
một ảnh, không phải 5 vật thể. Đối chiếu sang `detection_predictions.json` trên cùng ảnh đó thì model
detection tìm ra nhiều vật riêng biệt, trong đó có bus 0.912557 và một person 0.769676. Một nhãn cấp ảnh
buộc phải gộp cả cảnh đường phố đó lại thành một từ.

**3.** Danh sách tên lớp đi kèm taxonomy của checkpoint, model không tự nghĩ ra. File `yolo11n-cls.pt`
chứa sẵn bảng ánh xạ chỉ số sang tên lớp của ImageNet-1K; khi chạy, model xuất ra một dãy điểm số có độ
dài cố định, rồi notebook tra tên theo chỉ số. Cùng ảnh `traffic` này, hai checkpoint khác nhau cho hai
hệ tên khác nhau: `yolo11n-cls.pt` trả về cab theo ImageNet-1K, còn `yolo11n.pt` trả về bus và car theo
COCO-80. Ảnh không đổi, chỉ taxonomy đổi.

**4.** Vì `class_id` chỉ có nghĩa bên trong một taxonomy cụ thể. Trong output của bài này, `class_id` 468
là cab của ImageNet-1K, còn `class_id` 5 là bus của COCO-80; hai con số nằm ở hai hệ khác nhau nên không
so sánh trực tiếp được. Thiếu `taxonomy_name` thì về sau không ai giải nghĩa lại được con số đó. Nếu chỉ
ghi tên lớp thì gặp rủi ro khác: tên lớp có thể được viết khác giữa các phiên bản dữ liệu, hoặc hai
taxonomy dùng cùng một từ cho hai nghĩa. Ghi đủ ba trường thì vẫn đối chiếu được nhãn cũ với nhãn mới và
so sánh được giữa các lần chạy.

**5.** Guideline phải nói rõ quy tắc chọn một lớp khi ảnh có nhiều chủ thể, ít nhất là: nhãn mô tả toàn
cảnh hay mô tả vật thể nổi bật nhất; thứ tự ưu tiên khi có nhiều ứng viên, ví dụ ưu tiên vật chiếm diện
tích lớn nhất hoặc vật ở tiền cảnh; cách xử lý khi hai ứng viên gần bằng nhau; và trường hợp không lớp
nào trong danh sách khớp với ảnh thì phải escalation chứ không chọn lớp gần đúng. Ảnh `traffic` là ví dụ
đúng kiểu đó: nó có nhiều xe và có cả người, nên nếu guideline không đặt luật ưu tiên thì hai người gán
nhãn sẽ cho ra hai nhãn khác nhau mà không ai sai theo quy tắc.

**6.** `score` là con số do chính model sinh ra để xếp hạng các lớp, nó không đo mức độ phù hợp của nhãn
với guideline. Ở đây cab được 0.510915, nghĩa là model xếp nó cao nhất trong 1000 lớp ImageNet, chứ không
nghĩa là cab đã được người xác nhận là nhãn đúng cho ảnh này. Model cũng luôn phải chọn trong danh sách
lớp cố định của nó, nên khi ảnh chứa thứ không có trong taxonomy, model vẫn trả về lớp gần nhất kèm một
`score` trông có vẻ thuyết phục. Danh sách lớp của ImageNet-1K hướng về từng vật thể, không có lớp nào mô
tả một con phố có nhiều xe và có người, nên top-1 buộc phải rơi vào một loại xe. Ground truth là nhãn người gán theo guideline, nên nó phải được người xác nhận,
không phải bởi điểm số của model.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Record dưới đây lấy từ `sample_id = kitchen`.

**1.** `class_name` = person ; `score` = 0.912625 ; `bbox_xyxy` = [385.33, 69.24, 498.92, 348.92] ;
`bbox_width` = 113.58 ; `bbox_height` = 279.68 ; `coordinate_unit` = pixel ; `bbox_format` = xyxy.
Ảnh `kitchen` có `image_width` = 640 và `image_height` = 427.

**2.** Bốn số trong `bbox_xyxy` là toạ độ pixel, gốc `(0, 0)` nằm ở góc trên-trái của ảnh, trục x chạy
sang phải và trục y chạy xuống dưới. Cặp số đầu `[x_min, y_min]` là góc trên-trái của hộp: cách mép trái
ảnh `x_min` pixel và cách mép trên `y_min` pixel. Cặp số sau `[x_max, y_max]` là góc dưới-phải, đo theo
cùng một gốc. Với record ở câu 1, góc trên-trái của hộp nằm cách mép trái ảnh 385.33 pixel và cách mép
trên 69.24 pixel, góc dưới-phải cách mép trái 498.92 pixel và cách mép trên 348.92 pixel. Hộp rộng 113.58
pixel và cao 279.68 pixel, tức là người này chiếm khoảng 17.8% chiều rộng nhưng tới 65.5% chiều cao của
ảnh 640 x 427.

Có một chi tiết nhỏ nên ghi lại: 498.92 trừ 385.33 ra 113.59, trong khi `bbox_width` ghi 113.58. Lệch
0.01 là do notebook làm tròn riêng từng số trong `bbox_xyxy`, còn `bbox_width` thì tính từ giá trị gốc
chưa làm tròn rồi mới làm tròn kết quả. Hai trường không mâu thuẫn nhau, nhưng nó nhắc rằng không nên lấy
`bbox_width` để suy ngược ra toạ độ, và khi so nhãn giữa hai hệ thống thì phải biết con số mình đang đọc
được làm tròn ở bước nào.

**3.** Đếm trên ảnh `kitchen` theo ô so sánh ngưỡng: ngưỡng `0.20` cho 17 prediction, ngưỡng `0.35` cho 11
prediction, ngưỡng `0.60` cho 6 prediction. Ngưỡng cao hơn giữ lại ít prediction hơn, vì ngưỡng chỉ lọc
theo `score` chứ không nói gì về việc vật thể có thật trong ảnh hay không. Phần chênh lệch mới là chỗ đáng
đọc. Hạ từ `0.35` xuống `0.20` thêm đúng 6 prediction: 3 spoon, 1 bottle, 1 dining table và 1 potted plant;
riêng potted plant thì trong ảnh không có chậu cây nào, nhiều khả năng model đang bắt vào bó rau thơm treo
ở góc trên bên trái, nên đó là một nhãn cần người xem lại chứ không phải một vật thật mới xuất hiện. Nâng
từ `0.35` lên `0.60` bỏ đi 5 prediction: 2 cup và 3 bowl, đều là vật nhỏ hoặc bị che một phần trên mặt bàn.
Cùng một bức ảnh, ba con số 17 / 11 / 6 đều do ngưỡng quyết định, không do ảnh thay đổi.

**4.** Hạ ngưỡng thì reviewer phải xử lý thêm đúng phần hộp có điểm thấp, và đó là phần nhiều lỗi nhất.
Với mỗi hộp mới, reviewer phải chọn xoá nó, sửa lại cho khít, hay giữ nguyên. Hạ ngưỡng cũng hay sinh ra
hai hộp cùng đè lên một vật, nên phải soát thêm chuyện trùng hộp, và soát xem hộp mới có thực sự chứa vật
thuộc lớp trong phạm vi hay chỉ bắt vào nền. Số hộp cần duyệt tăng lên trong khi phần lớn rồi cũng bị
loại, và chỗ này mới đáng lo: reviewer mỏi rồi duyệt thiếu nhất quán.

**5.** Quy tắc hộp: hộp phải bao sát toàn bộ phần nhìn thấy được của vật thể, chạm vào pixel ngoài cùng ở
cả bốn phía, không chừa viền nền và không cắt mất phần nào đang nhìn thấy; nếu vật thể bị mép ảnh cắt thì
hộp dừng đúng ở mép ảnh, không ước lượng phần nằm ngoài ảnh. Quy tắc này dùng được cho mọi ảnh vì nó chỉ
nói về phần nhìn thấy, không phụ thuộc vào vật thể cụ thể nào.

**6.** Trên ảnh `kitchen`, vật bị che khuất mà tôi thấy là người thứ hai, được detect với `score` 0.61 ở
phía trái bàn: trong ảnh chỉ nhìn thấy bàn tay và một phần cánh tay đang đặt trên mặt bàn, phần còn lại
của người đó nằm ngoài khung ảnh. Người thứ nhất (`score` 0.91) đứng quay lưng ở giữa ảnh và nhìn thấy gần
như toàn thân, nên hai record cùng lớp `person` này có mức độ nhìn thấy rất khác nhau. Tôi xử lý bằng
guideline, vì "vật bị khung ảnh cắt thì gán nhãn theo phần nhìn thấy và hộp dừng ở mép ảnh" là quy tắc
dùng lại được cho mọi ảnh, không riêng ảnh này. Cách phân biệt tôi dùng: nếu tình huống chỉ cần một quy tắc dùng lại
được cho nhiều ảnh, ví dụ vật bị mép ảnh cắt thì gán nhãn theo phần nhìn thấy và hộp dừng ở mép, thì viết
thành quy tắc trong guideline. Nếu tình huống đòi thông tin mà bản thân ảnh không trả lời được, ví dụ
không rõ vật đó có thuộc lớp trong phạm vi hay không, hoặc không rõ nên tính là một vật hay hai, thì đưa
lên Lab Coach, vì tự quyết sẽ tạo ra nhãn không nhất quán với người khác.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Record dưới đây lấy từ `sample_id = kitchen`.

**1.** `instance_id` = kitchen-001 ; `class_name` = person ; `score` = 0.899318 ;
`polygon_point_count` = 348 ; 3 điểm đầu của `polygon_xy` = [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0]].

**2.** Polygon đi theo đường biên nhìn thấy của vật thể, nên nó loại được phần nền mà hộp buộc phải bao
vào. Với một vật mảnh và nằm chéo, hộp chứa rất nhiều pixel nền, còn polygon thì không. Số điểm đi theo
đúng độ phức tạp của biên: trên ảnh `kitchen`, `kitchen-010` (bowl, hộp chỉ khoảng 29 x 16 pixel) chỉ cần
12 điểm, `kitchen-002` (bowl) cần 67 điểm, `kitchen-001` (person) cần 348 điểm, còn `kitchen-005` (dining
table) cần tới 598 điểm vì biên của nó bị các vật trên mặt bàn cắt vụn. Trong khi đó mọi hộp đều chỉ có
đúng 4 số, bất kể vật là cái bát nhỏ hay cả cái bàn. Nhờ vậy polygon cho biết hình dạng và diện tích thật
của vật, và tách được hai vật nằm chồng nhau trong trường hợp hai hộp của chúng trùng nhau gần hết.

**3.** `instance_id` chỉ để phân biệt từng vật thể riêng lẻ bên trong bộ output của bài lab này. Nó không
phải class ID: `class_id` trả lời vật đó là gì và lặp lại ở mọi vật cùng lớp, còn `instance_id` trả lời
cái nào trong ảnh. Nó cũng không phải tracking ID: tracking ID phải giữ nguyên cho cùng một vật thể qua
nhiều khung hình của video, trong khi `instance_id` ở đây có dạng `<sample_id>-NNN` và được đánh theo thứ
tự record trong lần chạy này, nên chạy lại là có thể đổi số.

**4.** Quy tắc vẽ biên mask: vẽ theo đúng mép nhìn thấy của vật thể, đi vào các chỗ lõm thay vì cắt ngang,
và chừa lại những lỗ hở nhìn thấy nền phía sau; khi vật bị che thì chỉ vẽ phần đang nhìn thấy và không nội
suy phần bị che; hai vật cùng lớp mà chạm nhau thì vẫn vẽ hai polygon riêng, đường biên đặt đúng chỗ tiếp
giáp và không để polygon này liếm sang vật kia.

**5.** Trên ảnh `kitchen`, chỗ hai vật chạm nhau mà tôi thấy là hai instance cùng lớp `bowl`:
`kitchen-002` (`score` 0.735743, hộp [31.44, 343.77, 100.89, 385.55]) và `kitchen-006` (`score` 0.528202,
hộp [0.11, 302.85, 85.7, 370.15]). Hai hộp này đè lên nhau 1431 pixel vuông, tức khoảng 49% diện tích hộp
của `kitchen-002` nằm trong hộp của `kitchen-006`, IoU khoảng 0.198. Đây đúng là tình huống hộp không tách
được hai vật còn polygon thì tách được, và cũng là chỗ dễ sai nhất khi vẽ mask vì biên chung của hai cái
bát nằm ngay trong vùng chồng đó. Tôi xử lý bằng guideline: quy tắc "hai vật cùng lớp chạm nhau thì vẫn
vẽ hai polygon riêng, đường biên đặt đúng khe nhìn thấy giữa chúng" dùng lại được cho mọi ảnh. Chỉ khi
nhìn vào ảnh mà vẫn không xác định được khe giữa hai vật, hoặc không rõ đó là một cái bát hay hai cái xếp
chồng, thì mới escalation, vì lúc đó người gán nhãn đang đoán chứ không còn đọc ảnh nữa.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh, chọn trong danh sách cố định | Top-1 của `traffic` là cab (0.510915) theo ImageNet-1K, trong khi detection trên cùng ảnh đó tìm ra bus, car và person: một nhãn cấp ảnh phải gộp cả cảnh thành một từ | Chọn một lớp theo quy tắc ưu tiên trong guideline; không lớp nào khớp thì escalation | Nhãn có đúng quy tắc chủ thể nổi bật không, lớp có nằm trong taxonomy không |
| Phát hiện vật thể | Một lớp và một hộp khít cho mỗi vật thể nhìn thấy | Cùng một ảnh `kitchen` cho 17 / 11 / 6 prediction ở ngưỡng `0.20` / `0.35` / `0.60`; ở mức `0.20` còn xuất hiện potted plant trong khi ảnh không có chậu cây nào | Gán hộp cho mọi vật thuộc lớp trong phạm vi, kể cả vật bị che một phần | Có vật nào bị bỏ sót, hộp có khít không, có hộp trùng trên cùng một vật không |
| Instance segmentation | Một lớp và một polygon cho mỗi instance | Hai `bowl` chồng hộp lên nhau tới 49% (mục 3.5); và cùng ảnh, cùng ngưỡng `0.35`, detection với segmentation đều ra 11 prediction nhưng khác lớp: segmentation có thêm potted plant, dining table và 2 spoon, detection có thêm 2 cup, 1 bowl, 1 oven | Vẽ biên theo phần nhìn thấy, mỗi instance một polygon riêng | Số polygon có bằng số instance nhìn thấy không, biên ở chỗ tiếp giáp có đúng không |

Nối theo chuỗi `ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`: guideline quyết
định đơn vị nhãn ở cột thứ hai, ground truth là nhãn người gán theo guideline đó, model học từ ground truth
nên chỉ nói được bằng taxonomy nó đã học, còn các sai lệch ở cột thứ ba là việc của QC. Bài lab này đứng ở
mắt `prediction` rồi đọc ngược về phía trước, nên cái thu được là hình dạng mà ground truth phải có, không
phải kết luận ảnh có những gì.

## 5. An toàn dữ liệu

**1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì?**
Chỉ dùng ba ảnh công khai mà notebook đã ghim kèm checksum và
attribution, không đưa ảnh cá nhân, khuôn mặt, biển số, dữ liệu khách hàng hay dữ liệu nội bộ lên Colab và
GitHub công khai, và không ghi họ tên, MSSV, email hay số điện thoại vào báo cáo cũng như vào output.

**2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?**
Nếu gặp ảnh hoặc dữ liệu ngoài phạm vi thì tôi dừng lại, không chạy tiếp và không upload, rồi báo
Lab Coach hoặc giảng viên phụ trách qua kênh hỗ trợ của lớp. Nếu đã kịp đẩy lên repo công khai thì báo
ngay để xử lý. Tôi không gửi password, token GitHub hay mã xác thực Google cho bất kỳ ai.

## 6. Checklist trước khi nộp link repo

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên VLearn

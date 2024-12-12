# Step 1:  Vô hiệu hóa kho lưu trữ từ CD-ROM
```bash
sudo nano /etc/apt/sources.list
```

## Tìm dòng có chứa cdrom://: Tìm dòng bắt đầu với cdrom://. Nó có thể trông như sau:
```bash
deb cdrom:[Ubuntu-Server 18.04 LTS _Bionic Beaver_ - Release amd64 (20180426)]/ bionic main restricted
```

## Vô hiệu hóa dòng này: Bạn có thể vô hiệu hóa dòng này bằng cách thêm dấu # ở đầu, để nó trông như sau:
```bash
# deb cdrom:[Ubuntu-Server 18.04 LTS _Bionic Beaver_ - Release amd64 (20180426)]/ bionic main restricted
```

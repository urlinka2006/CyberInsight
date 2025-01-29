import os
import hashlib
import paramiko
from scp import SCPClient
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Функція для обчислення SHA256 хешу
def calculate_hash(file_path):
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logging.error(f"Помилка під час обчислення хешу для {file_path}: {e}")
        return None

# Функція передачі файлу через SCP
def transfer_file_to_caine(local_path, remote_path, hostname, username, password):
    try:
        # Налаштування з'єднання через SSH
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password)

        # Використання SCP для передачі файлу
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(local_path, remote_path)
        logging.info(f"Файл {local_path} успішно передано на {hostname}:{remote_path}")
        ssh.close()
    except Exception as e:
        logging.error(f"Помилка під час передачі файлу: {e}")

# Основна функція
def main():
    # Шлях до дампу пам'яті
    dump_file = "TOTAL_RECALL_memory_forensics_CHALLENGE/SECURITYNIK-WIN-20231116-235706.dmp" # Шлях до дампу памʼяті в цільовій машині
    caine_path = "/mnt/new_disk/"  # Директорія на CAINE, куди передаємо файл памʼяті

    # Дані для з'єднання з CAINE
    hostname = "192.168.0.102"  # IP-адреса CAINE
    username = "caine"          # Ім'я користувача в CAINE
    password = "12345678"  # Пароль для підключення до CAINE

    # Перевірка існування файлу
    if not os.path.exists(dump_file):
        logging.error(f"Файл {dump_file} не знайдено!")
        return

    # Обчислення хешу
    hash_before = calculate_hash(dump_file)
    if not hash_before:
        logging.error("Не вдалося обчислити хеш. Завершення роботи.")
        return

    logging.info(f"SHA256 хеш файлу перед передачею: {hash_before}")

    # Передача файлу
    transfer_file_to_caine(dump_file, os.path.join(caine_path, os.path.basename(dump_file)), hostname, username, password)

    # Додаткова перевірка хешу після передачі
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password)

        remote_file_path = os.path.join(caine_path, os.path.basename(dump_file))
        stdin, stdout, stderr = ssh.exec_command(f"sha256sum {remote_file_path}")
        hash_after = stdout.read().decode().split()[0]
        ssh.close()

        if hash_before == hash_after:
            logging.info("Цілісність файлу підтверджена.")
        else:
            logging.warning("Хеші не збігаються! Файл пошкоджено.")
    except Exception as e:
        logging.error(f"Помилка під час перевірки хешу на CAINE: {e}")

if __name__ == "__main__":
    main()
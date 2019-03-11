import time
import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials


class Logger:
    def __init__(self, filename, stdout=None):
        self.stdout = stdout
        self.logfile = open(filename, 'w')

    def print(self, text):
        if self.stdout is not None:
            self.stdout.write(text)
        self.logfile.write(text)

    def close(self):
        self.logfile.close()


class GoogleSheetLogger:
    def __init__(self, credential_file, workbook_name="experiments_logs"):
        # use creds to create a client to interact with the Google Drive API
        scope = ['https://spreadsheets.google.com/feeds',
                 'https://www.googleapis.com/auth/drive']
        creds = ServiceAccountCredentials.from_json_keyfile_name(credential_file, scope)
        client = gspread.authorize(creds)

        # Find a workbook by name.
        self.workbook = client.open(workbook_name)

    def append_row(self, values, sheet_name):
        sheet = self.workbook.worksheet(sheet_name)
        header = sheet.row_values(1)

        if isinstance(values, list):
            sheet.append_row(values)
        elif isinstance(values, dict):
            values_list = [None for _ in header]
            for i, header in enumerate(header):
                if header in values:
                    values_list[i] = values[header][0]
                    if len(values[header]) > 1 and header[i+1] == '':
                        values_list[i+1] = values[header][1]
                elif header == "Timestamp":
                    # automatically add timestamp
                    values_list[i] = get_formatted_timestamp()

            sheet.append_row(values_list)


def get_model_dir_timestamp(prefix="", suffix="", connector="_"):
    """
    Creates a directory name based on timestamp.

    Args:
        prefix:
        suffix:
        connector: one connector character between prefix, timestamp and suffix.

    Returns:

    """
    return prefix+connector+str(int(time.time()))+connector+suffix


def get_formatted_timestamp():
    """Returns the current local time as nicely formatted string."""
    return str(datetime.datetime.now()).split('.')[0]

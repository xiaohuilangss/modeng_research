# encoding=utf-8

"""
有关email类的实现
"""
import email
import json
import os, pickle
import poplib
import smtplib
import mimetypes
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.encoders import encode_base64
from email.parser import Parser
from email.utils import parseaddr

import time

from my_config.GlobalSetting import root_path
from my_config.log import MyLog
from sdk.MyTimeOPT import get_current_datetime_str

logger = MyLog('email_class').logger
logger_eml = MyLog('email_class_eml').logger

class MyEmail:
    def __init__(self, sender_info_json_url, modeng=False):
        
        self.sender_info_json_url = sender_info_json_url
        self.modeng = modeng
        self.sender = ''
        self.passwd = ''
        self.smtp_domain = ''  # 'smtp.163.com'
        
        self.config_sender_info_by_json()
    
    def config_sender_info_by_json(self):
        """
        加载含有发件人账密的json文件，读取账密
        json文件有以下字段：
        sender：发件人账号
        passwd：发件人密码
        smtp_domain：发件邮箱服务器域名
        :return:
        """
        if not os.path.exists(self.sender_info_json_url):
            logger.error('在路径【%s】没有找到包含发件人账密的json文件！' % self.sender_info_json_url)
            return False
        
        try:
            with open(self.sender_info_json_url, 'r') as f:
                data_json = json.load(f)
                self.sender = data_json['sender']
                self.passwd = data_json['passwd']
                self.smtp_domain = data_json['smtp']
            return True
        
        except Exception as e_:
            logger.error('从路径【%s】读取发件人账密时出现异常错误！:\n %s' % (self.sender_info_json_url, str(e_)))
            return False
    
    @staticmethod
    def get_attachment(attachment_file_path):
        
        """
        获取附件
        :param attachment_file_path: 附件文件路径
        :return:
        """
        
        # 根据 guess_type方法判断文件的类型和编码方式
        content_type, encoding = mimetypes.guess_type(attachment_file_path)
        
        # 如果根据文件的名字/后缀识别不出是什么文件类型,使用默认类型
        if content_type is None or encoding is not None:
            content_type = 'application/octet-stream'
        
        # 根据contentType 判断主类型与子类型
        main_type, sub_type = content_type.split('/', 1)
        file = open(attachment_file_path, 'rb')
        
        # 根据主类型不同，调用不同的文件读取方法
        if main_type == 'text':
            attachment = MIMEBase(main_type, sub_type)
            attachment.set_payload(file.read())
            encode_base64(attachment)
        
        elif main_type == 'message':
            attachment = email.message_from_file(file)
        
        elif main_type == 'image':
            attachment = MIMEImage(file.read())
        
        # elif mainType == 'audio':  # 音频
        # attachment = MIMEAudio(file.read(), _subType=subType)
        
        else:
            attachment = MIMEBase(main_type, sub_type)
            attachment.set_payload(file.read())
            encode_base64(attachment)
        
        file.close()
        """
            Content-disposition 是 MIME 协议的扩展,
            MIME 协议指示 MIME 用户代理如何显示附加的文件。
            Content-disposition其实可以控制用户请求所得的内容存为一个文件的时候提供一个默认的文件名，
            文件直接在浏览器上显示或者在访问时弹出文件下载对话框。
            content-disposition = "Content-Disposition" ":" disposition-type *( ";" disposition-parm )。
            # Content-Disposition为属性名 disposition-type是以什么方式下载，
            如attachment为以附件方式下载 disposition-parm为默认保存时的文件名
        """
        attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_file_path))
        
        return attachment
    
    def connect_smtp(self):
        try:
            if self.modeng:
                mail_server = smtplib.SMTP_SSL(self.smtp_domain, port=465)
                # mail_server.ehlo()
                # mail_server.starttls()
                mail_server.ehlo()
                mail_server.login(self.sender, self.passwd)
                return mail_server
            else:
                mail_server = smtplib.SMTP(self.smtp_domain)
                mail_server.ehlo()
                mail_server.starttls()
                mail_server.ehlo()
                mail_server.login(self.sender, self.passwd)
                return mail_server
        except Exception as e_:
            logger.exception('连接smtp失败，原因：\n %s' % str(e_))
            return None
    
    def send_email(self,  *attachment_file_paths, subject, recipient, text, send_type='To', text_type='plain'):
        """
        发送邮件函数：参数（邮件主题，邮件内容，邮件附件（可多选））
        :param send_type:
        :param text_type:
        :param subject:
        :param recipient:
        :param text:
        :param attachment_file_paths:
        :return:
        """
        
        # 发送附件时需调用 MIMEMultipart类，创建 MIMEMultipart,并添加信息头
        msg = MIMEMultipart()
        
        """
        MIME邮件的基本信息、格式信息、编码方式等重要内容都记录在邮件内的各种域中，
        域的基本格式：{域名}：{内容}，域由域名后面跟“：”再加上域的信息内容构成，
        一条域在邮件中占一行或者多行，
        域的首行左侧不能有空白字符，比如空格或者制表符，
        占用多行的域其后续行则必须以空白字符开头。
        域的信息内容中还可以包含属性，属性之间以“;”分隔，
        属性的格式如下：{属性名称}=”{属性值}”。
        """
        
        # 登记基础信息及正文
        msg['From'] = self.sender
        msg[send_type] = ";".join(recipient)
        msg['Subject'] = subject
        msg.attach(MIMEText(text, _subtype=text_type))
        
        # 添加附件
        if len(attachment_file_paths) == 1:
            if (
                    isinstance(attachment_file_paths[0], type(())) |
                    isinstance(attachment_file_paths[0], type([]))
            ):
                attachment_file_paths = attachment_file_paths[0]
        
        for attachmentFilePath in attachment_file_paths:
            msg.attach(self.get_attachment(attachmentFilePath))
        
        try:
            mail_server = self.connect_smtp()
            if isinstance(mail_server, type(None)):
                logger.error('连接smtp服务器失败，返回！')
                return
            
            mail_server.sendmail(self.sender, recipient, msg.as_string())
            mail_server.close()
            logger.info('成功向【%s】发送邮件！' % recipient)
        
        except Exception as e_:
            logger_eml.error('发件失败，原因：\n %s' % str(e_))


class EmailRead(MyEmail):
    def __init__(self, sender_info_json_url, modeng=False):
        super().__init__(sender_info_json_url=sender_info_json_url, modeng=modeng)
        """
        读邮件类
        用完一定记得执行self.disconnect()函数，退出连接！
        """
        self.msg_count = 0
        self.mailbox_size = 0
        self.mail_server = None
        self.update_mailbox_info()
    
    def connect(self):
        try:
            """ 创建POP3对象，添加用户名和密码"""
            self.mail_server = poplib.POP3(self.smtp_domain)
            self.mail_server.user(self.sender)
            self.mail_server.pass_(self.passwd)
            logger.debug('完成mailbox连接操作！')
        except Exception as e_:
            logger_eml.exception('连接邮箱账户出错！具体：\n%s' % str(e_))
    
    def update_mailbox_info(self):
        """
        获取当前邮箱状况，邮件数等
        :return:
        """
        self.connect()
        
        """获取邮件数量和占用空间"""
        self.msg_count, self.mailbox_size = self.mail_server.stat()
        
        """获取邮件请求返回状态码、每封邮件的字节大小(b'第几封邮件 此邮件字节大小')"""
        # response, msgNumOctets, octets = self.mail_server.list()
    
    def disconnect(self):
        self.mail_server.quit()
    
    def get_mail_object(self, mail_num):
        """
        获取邮件对象
        :return:
        """
        
        # 获取邮件的信息
        response, msg_lines, octets = self.mail_server.retr(mail_num)
        
        # msgLines中为该邮件的每行数据,先将内容连接成字符串，再转化为email.message.Message对象
        msg_lines_to_str = b"\r\n".join(msg_lines).decode("utf8", "ignore")
        return Parser().parsestr(msg_lines_to_str)
    
    def get_mail_subject(self, msg_obj):
        
        msg_header = msg_obj["Subject"]
        
        # 对头文件进行解码
        return self.decode_msg_header(msg_header)
    
    @staticmethod
    def get_mail_date(msg_obj):
        return msg_obj["date"]
    
    @staticmethod
    def decode_msg_header(header):
        """
        解码头文件
        :param header: 需解码的内容
        :return:
        """
        value, charset = decode_header(header)[0]
        if charset:
            value = value.decode(charset)
        return value
    
    def get_sender_info(self, msg_obj):
        
        sender_content = msg_obj["From"]
        
        # parseaddr()函数返回的是一个元组(realname, emailAddress)
        sender_real_name, sender_adr = parseaddr(sender_content)
        
        # 将加密的名称进行解码
        sender_real_name = self.decode_msg_header(sender_real_name)
        
        return sender_adr, sender_real_name
    
    def get_mail_content(self, msg_obj):
        
        """获取邮件正文内容"""
        msg_body_contents = []
        if msg_obj.is_multipart():  # 判断邮件是否由多个部分构成
            msg_parts = msg_obj.get_payload()  # 获取邮件附载部分
            for messagePart in msg_parts:
                body_content = self.decode_body(messagePart)
                if body_content:
                    msg_body_contents.append(body_content)
        else:
            body_content = self.decode_body(msg_obj)
            if body_content:
                msg_body_contents.append(body_content)
        return msg_body_contents
    
    @staticmethod
    def decode_body(msg_part):
        """
        解码内容
       :param msg_part: 邮件某部分
        """
        content_type = msg_part.get_content_type()  # 判断邮件内容的类型,text/html
        text_content = ""
        if content_type == 'text/plain' or content_type == 'text/html':
            content = msg_part.get_payload(decode=True)
            charset = msg_part.get_charset()
            if charset is None:
                content_type = msg_part.get('Content-Type', '').lower()
                position = content_type.find('charset=')
                if position >= 0:
                    charset = content_type[position + 8:].strip()
            if charset:
                text_content = content.decode(charset)
        return text_content
    
    def get_mail_attach(self, msg_obj):
        """
        获取附件
        :return:
        """
        msg_attachments = []
        if msg_obj.is_multipart():  # 判断邮件是否由多个部分构成
            msg_parts = msg_obj.get_payload()  # 获取邮件附载部分
            for msg_part in msg_parts:
                name = msg_part.get_param("name")  # 名字存在，则表示此部分为附件
                if name:
                    file_name = self.decode_msg_header(name)  # 解码
                    msg_attachments.append(file_name)
        else:
            name = msg_obj.get_param("name")
            if name:
                file_name = self.decode_msg_header(name)
                msg_attachments.append(file_name)
        print(msg_attachments)
    
    def clear_mailbox(self):
        try:
            logger.debug('邮箱清空前有邮件【%d】封' % self.msg_count)
            for i in range(self.msg_count):
                self.mail_server.dele(i + 1)
            logger.debug('邮箱已清空！清空后邮箱状态：%s' % str(self.mail_server.stat()))
        
        except Exception as e_:
            logger.exception('清空邮箱时出错，原因：\n%s' % str(e_))
    
    def get_mail_obj_list(self):
        """
        返回邮箱对象及编号，编号用于定点删除
        :return:
        """
        return [(self.get_mail_object(x + 1), x + 1) for x in range(self.msg_count)]


if __name__ == '__main__':
    
    er = EmailRead(root_path + '/server/mail_subscribe/global_value/senderInfo.json')
    
    while True:
        er.update_mailbox_info()
        if er.msg_count > 0:
            obj_list = er.get_mail_obj_list()
            for obj in obj_list:
                print(er.get_mail_content(obj))
            er.clear_mailbox()
        else:
            logger.debug('【%s】：邮箱为空现在！' % get_current_datetime_str())
        
        time.sleep(10)



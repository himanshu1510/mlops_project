#!/usr/bin/env python
# coding: utf-8

# In[1]:


import smtplib, ssl


# In[5]:


port = 465
smtp_server = "smtp.gmail.com"
sender_email = "himanshu1507agrawal@gmail.com"
reciver_email = "agrawal1507himanshvinod@gmail.com"
password = "Shinu@456"
message = """Subject: Hi there

your model is trained with accuracy greater than 90%."""

context = ssl.create_default_context()
with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(sender_email, reciver_email, message)


# In[ ]:





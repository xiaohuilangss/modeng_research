# encoding=utf-8
from apscheduler.schedulers.blocking import BlockingScheduler


class Scheduler:
    def __init__(self):
        self.aps = BlockingScheduler()

    def add_job(self, func, trigger='cron', day_of_week='mon-fri', hour=16, minute=0):
        self.aps.add_job(
            func=func,
            trigger=trigger,
            day_of_week=day_of_week,
            hour=hour,
            minute=minute,
            misfire_grace_time=3600,
            coalesce=True)

    def timer_start(self):
        """
        需要被重写
        :return:
        """
        self.aps.start()
    
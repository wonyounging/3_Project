from datetime import datetime
from dateutil.relativedelta import *


class FindAnswer:
    def __init__(self, db, lsts):
        self.db = db
        self.lsts = lsts

    def recent_day(self):
        now = datetime.now()
        six_months_ago = now - relativedelta(months=6)
        recent = six_months_ago.strftime("%Y-%m-%d")
        return recent

    def time_sort(self, times):
        sort_time = [[], [], [], [], []]
        return_time = []

        for time in times:
            if "월" in time or "화" in time or "수" in time or "목" in time or "금" in time or "토" in time or "일" in time:
                if "요일" in time:
                    sort_time[4] = time
                elif "일" in time:
                    sort_time[1] = time
                else:
                    sort_time[0] = time
            elif "시" in time:
                sort_time[2] = time
            elif "분" in time:
                sort_time[3] = time

        for st in sort_time:
            if len(st) > 0:
                return_time.append(st)
        return return_time

    def intent_query(self):
        if self.lsts[0] == "추천":
            sql_intent = ' order by rand() limit 1;'  # 랜덤으로 하나 추천
        else:
            sql_intent = ''
        return sql_intent

    def emotion_query(self):
        if self.lsts[1] != "없음":
            if self.lsts[2] != "부정":
                sql_keyword = f"keyword = '{self.lsts[1]}'"
            else:
                sql_keyword = f"keyword != '{self.lsts[1]}'"
        else:
            sql_keyword = ''
        return sql_keyword

    def trend_query(self):
        if self.lsts[3] == '최신':
            sql_trend = f"opendate >= '{self.recent_day()} and people >= 1000000'"  # 최신(6개월)
        elif self.lsts[3] == '인기':
            sql_trend = "people >= 5000000"
        else:
            sql_trend = "people >= 1000000"
        return sql_trend

    def ner_query(self):
        sql_lst = []
        lst = self.lsts[4]

        # 배우 포함
        if lst[1] != []:
            for act in lst[1]:
                sql = f"actors like '%{act}%'"
                sql_lst.append(sql)

        # 장르 포함
        if lst[2] != []:
            for gen in lst[2]:
                sql = f"genre like '%{gen}%'"
                sql_lst.append(sql)

        # 국적 - 대표국적으로 구분
        if lst[3] != []:
            한국 = ['대한민국', '우리나라', '국내']
            외국 = ['해외', '외국']
            for nat in lst[3]:
                if nat in 한국:
                    sql = "repnation like '한국'"
                elif nat in 외국:
                    sql = "repnation not like '한국'"
                else:
                    sql = f"repnation like '{nat}'"
                sql_lst.append(sql)

        # 감독 포함
        if lst[4] != []:
            for direc in lst[4]:
                sql = f"director like '%{direc}%'"
                sql_lst.append(sql)

        if len(sql_lst) == 0:
            sql = ''
        else:
            sql = ' and '.join(sql_lst)
        return sql

    def final_query(self):
        lst = []
        lst.append(self.emotion_query())
        lst.append(self.trend_query())
        lst.append(self.ner_query())

        final_lst = []
        for query in lst:
            if query != '':
                final_lst.append(query)

        sql = ' and '.join(final_lst)
        final_sql = "select * from chat_movie where " + sql + self.intent_query()
        return final_sql

    def find_answer(self):
        title = self.lsts[4][0]
        time = self.time_sort(self.lsts[4][5])

        if self.lsts[0] == '추천':
            self.db.connect()
            find_dict = self.db.select_all(self.final_query())
            self.db.close()

            if len(find_dict) == 0:
                ans = "조건에 맞는 영화목록이 없습니다."
                return ans
            else:
                if find_dict[0]['poster'] == '-':
                    poster = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fi.pinimg.com%2Foriginals%2F04%2Fd4%2Fb9%2F04d4b9418e82a065f5542d0260b3d717.jpg&type=sc960_832"
                else:
                    poster = find_dict[0]['poster']
                ans = f'''
                    <table>
                    <tr><th colspan="2">영화 "{find_dict[0]['title']}" 추천드립니다:)</th></tr>
                    <tr>
                        <td rowspan="5"><img src="{poster}" alt="" style="width:110px;height:150px"></td>
                        <td> 영화명 : {find_dict[0]['title']}</td>
                    </tr>
                    <tr><td> 개봉일자 : {str(find_dict[0]['opendate'])[:10]}</td></tr>
                    <tr><td> 장르 : {find_dict[0]['genre']}</td></tr>
                    <tr><td> 대표국적 : {find_dict[0]['repnation']}</td></tr>
                    <tr><td> 감독 : {find_dict[0]['director']}</td></tr>
                    </table> '''
                return ans

        elif self.lsts[0] == '후기':
            if title == []:
                ans = "영화명과 같이 다시 입력해주세요."
                return ans

            else:
                self.db.connect()
                find_dict = self.db.select_all(
                    f'select * from chat_movie where title = "{title[0]}" order by people DESC')
                self.db.close()
                if len(find_dict) == 0:
                    ans = "해당 영화 후기가 없습니다."
                    return ans
                elif find_dict[0]['totscore'] == 'null':
                    ans = "해당 영화 후기가 없습니다."
                    return ans
                else:
                    if find_dict[0]['review'] != 'null':
                        code = "<tr><td> 평점</td><td> 리뷰</td></tr>"
                        for i in range(len(find_dict[0]['revscore'].split(' / '))):
                            code += f'''
                            <tr><td> {find_dict[0]['revscore'].split(' / ')[i]}</td>
                            <td>{find_dict[0]['review'].split(' / ')[i]}</td></tr>'''
                    else:
                        code = ''

                    ans = f'''
                            <table>
                            <tr><th colspan="2">영화 "{find_dict[0]['title']}"의 관람객 전체 평점은 {find_dict[0]['totscore']}점 입니다.</th></tr>
                            {code}
                            </table> '''
                    return ans

        elif self.lsts[0] == '예매':
            if len(self.lsts[4][0]) > 0 and len(self.lsts[4][5]) > 0:
                ans = f'''영화 "{title[0]}"이(가) {' '.join(time)}에 예약되었습니다.'''
            else:
                ans = f"예매하고 싶은 영화명과 <br> 예매 시간을 포함해서 다시 문의주세요."
            return ans

        elif self.lsts[0] == '정보':
            if title == []:
                ans = "영화명을 입력하세요."
                return ans

            else:
                self.db.connect()
                find_dict = self.db.select_all(
                    f'select * from chat_movie where title = "{title[0]}" order by people DESC')
                self.db.close()
                if find_dict[0]['poster'] == '-':
                    poster = "https://search.pstatic.net/sunny/?src=https%3A%2F%2Fi.pinimg.com%2Foriginals%2F04%2Fd4%2Fb9%2F04d4b9418e82a065f5542d0260b3d717.jpg&type=sc960_832"
                else:
                    poster = find_dict[0]['poster']

                ans = f'''
                            <table>
                            <tr><th colspan="2">영화 "{find_dict[0]['title']}" 정보입니다:)</th></tr>
                            <tr>
                                <td rowspan="5"><img src="{poster}" alt="" style="width:110px;height:150px"></td>
                                <td> 영화명 : {find_dict[0]['title']}</td>
                            </tr>
                            <tr><td> 개봉일자 : {str(find_dict[0]['opendate'])[:10]}</td></tr>
                            <tr><td> 장르 : {find_dict[0]['genre']}</td></tr>
                            <tr><td> 대표국적 : {find_dict[0]['repnation']}</td></tr>
                            <tr><td> 감독 : {find_dict[0]['director']}</td></tr>
                            </table> '''
                return ans
        else:
            ans = '죄송합니다. 다시 이용해주세요.'
            return ans
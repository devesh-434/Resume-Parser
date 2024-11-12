from bs4 import BeautifulSoup
import requests

def get_job_listings(category):
    joblist = []  # Local list to store job listings
    
    if isinstance(category, str):
        # Construct the URL based on the category string
        header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'}
        url = f'https://www.postjobfree.com/jobs?q={category}&l=India&radius=25'
        r = requests.get(url, headers=header)
        
        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(r.content, 'lxml')
        
        divs = soup.find_all('div', class_='snippetPadding')
        for item in divs:
            title = item.find('a').text.strip()
            company = item.find('span', class_='colorCompany').text.strip()
            job = {
                'title': title,
                'company': company
            }
            joblist.append(job)
    else:
        print("Category is not a string:", category)

    return joblist

document.addEventListener('DOMContentLoaded', function() {
    const table = document.querySelector('table');

    // 테이블 정렬
    table.querySelectorAll('th').forEach(header => {
        header.addEventListener('click', () => {
            const index = Array.from(header.parentElement.children).indexOf(header);
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            rows.sort((rowA, rowB) => {
                const cellA = rowA.children[index].innerText;
                const cellB = rowB.children[index].innerText;
                return cellA.localeCompare(cellB);
            });
            rows.forEach(row => table.querySelector('tbody').appendChild(row));
        });
    });

    // 검색 기능
    const searchInput = document.createElement('input');
    searchInput.setAttribute('type', 'text');
    searchInput.setAttribute('placeholder', 'Search...');
    document.body.insertBefore(searchInput, table);

    searchInput.addEventListener('input', function() {
        const filter = this.value.toLowerCase();
        table.querySelectorAll('tbody tr').forEach(row => {
            const text = row.innerText.toLowerCase();
            row.style.display = text.includes(filter) ? '' : 'none';
        });
    });

    // 버튼 클릭 시 그래프 팝업 열기
    table.querySelectorAll('.state-button').forEach(button => {
        button.addEventListener('click', function() {
            const tenure = this.getAttribute('data-tenure');
            const totalCharges = this.getAttribute('data-totalcharges');
            showGraph(tenure, totalCharges);
        });
    });

    // 그래프 팝업 함수
    function showGraph(tenure, totalCharges) {
        // 팝업을 위한 div 생성
        const popup = document.createElement('div');
        popup.classList.add('graph-popup');
        popup.style.position = 'fixed';
        popup.style.top = '50%';
        popup.style.left = '50%';
        popup.style.transform = 'translate(-50%, -50%)';
        popup.style.backgroundColor = 'white';
        popup.style.padding = '20px';
        popup.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
        popup.style.zIndex = '1000';

        // 그래프 제목 추가
        const graphTitle = document.createElement('h2');
        graphTitle.innerText = `Tenure: ${tenure}, Total Charges: ${totalCharges}`;
        popup.appendChild(graphTitle);

        // Close 버튼 추가
        const closeButton = document.createElement('button');
        closeButton.innerText = 'Close';
        closeButton.addEventListener('click', () => {
            document.body.removeChild(popup);
        });
        popup.appendChild(closeButton);

        // 그래프 그리기 위한 canvas 추가
        const canvas = document.createElement('canvas');
        canvas.width = 400; // 원하는 너비
        canvas.height = 200; // 원하는 높이
        popup.appendChild(canvas);
        document.body.appendChild(popup);

        // 그래프 그리기 예시 (여기에 실제 그래프 그리는 코드 추가)
        const ctx = canvas.getContext('2d');
        const chartData = {
            labels: ['Label1', 'Label2', 'Label3'], // x축 레이블
            datasets: [{
                label: 'Churn Data',
                data: [10, 20, 30], // y축 데이터 (예시)
                borderColor: 'red',
                fill: false
            }]
        };

        new Chart(ctx, {
            type: 'line', // 또는 'bar', 'pie' 등
            data: chartData,
            options: {
                responsive: true,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Tenure'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Total Charges'
                        }
                    }
                }
            }});
    }
});

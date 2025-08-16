import { Line } from 'react-chartjs-2';
import { Chart, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale } from 'chart.js';

Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale);

function PriceChart() {
  const data = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    datasets: [
      {
        label: 'Portfolio Value',
        data: [25000, 25400, 25250, 25750, 25900],
        borderColor: 'rgb(37,99,235)',
        backgroundColor: 'rgb(37,99,235,0.2)',
        fill: true,
        tension: 0.1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Portfolio Value This Week' },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6 mb-8 max-w-3xl w-full mx-auto">
      <Line data={data} options={options} />
    </div>
  );

}

export default PriceChart;

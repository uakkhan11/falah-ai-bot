import { Line } from 'react-chartjs-2';
import { Chart, LineController, LineElement, PointElement, LinearScale, Title, CategoryScale } from 'chart.js';

// Register necessary chart modules!
Chart.register(LineController, LineElement, PointElement, LinearScale, Title, CategoryScale);

function PriceChart() {
  // Chart data
  const data = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
    datasets: [
      {
        label: 'Portfolio Value',
        data: [25000, 25400, 25250, 25750, 25900],
        borderColor: 'rgb(37,99,235)',
        backgroundColor: 'rgba(37,99,235,0.2)',
        fill: true,
        tension: 0.1,
      },
    ],
  };

  // Chart options -- customize here!
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    aspectRatio: 2, // Change this ratio or remove for default
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: 'Portfolio Value This Week' },
    },
  };

  // Control chart/container size here
  return (
    <div
      className="bg-white rounded-lg shadow-md p-6 mb-8 max-w-xl w-full mx-auto"
      style={{ height: '300px' }}
    >
      <Line data={data} options={options} />
    </div>
  );
}

export default PriceChart;

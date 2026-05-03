import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Layout from './components/Layout'
import QueryPage from './pages/QueryPage'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Layout>
        <QueryPage />
      </Layout>
    </QueryClientProvider>
  )
}


import React, {useState, useEffect} from "react";
import Table from 'react-bootstrap/Table';

const Userall = () => {
    const [users, setUsers] = useState([])

    useEffect(() => {
        const fetchData = async () => {
            const url = "http://localhost:5000/api/users"
            const response = await fetch(url)
            const data = await response.json()
            
            setUsers(data.users)
        }
        
        fetchData()
    }, [users])
    
    const CancelDataHandler = async (id) => {
        const url = 'http://localhost:5000/api/users/' + id
        await fetch(url, { method: 'DELETE' })

        const url2 = "http://localhost:5000/api/users"
        const response = await fetch(url2)
        const data = await response.json()
        setUsers(data.users)
    }

    const UpdateDataHandler = async (id) => {

    }

    return (
        <Table striped center bordered hover responsive size="sm">
            <thead centre>USERS LIST</thead>
            {users.map((obj, idx) => ( 
                              
                <tr>
                    <th>{obj.userId}</th>
                    <td>{obj.name}</td>
                    <td>{obj.password}</td>
                    <td><button onClick={() => CancelDataHandler(obj.userId)} type="submit">Cancel</button></td>
                    <td><button >Update</button></td>
                </tr>
            ))}
        </Table>
    )
}

export default Userall;